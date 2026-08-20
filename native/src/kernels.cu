#include "pvr/kernels.hpp"

#include "pvr/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace pvr {
namespace {

__constant__ ImageView kImages[128];
__constant__ QuadView kQuads[128];

__device__ float cubic_weight(float value) {
  constexpr float coefficient = -0.75F;  // OpenCV INTER_CUBIC
  value = fabsf(value);
  if (value <= 1.0F) {
    return (coefficient + 2.0F) * value * value * value -
           (coefficient + 3.0F) * value * value + 1.0F;
  }
  if (value < 2.0F) {
    return coefficient * value * value * value -
           5.0F * coefficient * value * value +
           8.0F * coefficient * value - 4.0F * coefficient;
  }
  return 0.0F;
}

__global__ void resize_normalize_kernel(float* output, int count, int out_w,
                                        int out_h, float3 mean,
                                        float3 inverse_std, float scale,
                                        bool swap_red_blue, ResizeMode mode,
                                        Interpolation interpolation,
                                        float pad_value) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int image_index = blockIdx.z;
  if (x >= out_w || y >= out_h || image_index >= count) {
    return;
  }
  const auto image = kImages[image_index];
  int resized_width = out_w;
  int resized_height = out_h;
  int left = 0;
  int top = 0;
  if (mode == ResizeMode::letterbox_center) {
    const float ratio = min(static_cast<float>(out_w) / image.width,
                            static_cast<float>(out_h) / image.height);
    resized_width = max(1, static_cast<int>(roundf(image.width * ratio)));
    resized_height = max(1, static_cast<int>(roundf(image.height * ratio)));
    left = (out_w - resized_width) / 2;
    top = (out_h - resized_height) / 2;
  } else if (mode == ResizeMode::fit_height_left) {
    resized_width = min(
        out_w, max(1, static_cast<int>(ceilf(
                       out_h * static_cast<float>(image.width) / image.height))));
  }
  const int local_x = x - left;
  const int local_y = y - top;
  const bool padded = local_x < 0 || local_x >= resized_width || local_y < 0 ||
                      local_y >= resized_height;
  const float source_x =
      (static_cast<float>(local_x) + 0.5F) * image.width / resized_width - 0.5F;
  const float source_y =
      (static_cast<float>(local_y) + 0.5F) * image.height / resized_height - 0.5F;
  const int source_x0 = static_cast<int>(floorf(source_x));
  const int source_y0 = static_cast<int>(floorf(source_y));
  const int x0 = max(0, min(image.width - 1, source_x0));
  const int y0 = max(0, min(image.height - 1, source_y0));
  const int x1 = max(0, min(image.width - 1, source_x0 + 1));
  const int y1 = max(0, min(image.height - 1, source_y0 + 1));
  const float wx = source_x - source_x0;
  const float wy = source_y - source_y0;
  const auto* row0 = image.pixels + static_cast<std::size_t>(y0) * image.pitch;
  const auto* row1 = image.pixels + static_cast<std::size_t>(y1) * image.pitch;
  const std::size_t plane = static_cast<std::size_t>(out_w) * out_h;
  float values[3];
  for (int channel = 0; channel < 3; ++channel) {
    if (padded) {
      values[channel] = pad_value * scale;
    } else if (interpolation == Interpolation::cubic) {
      float value = 0.0F;
      const int origin_x = static_cast<int>(floorf(source_x));
      const int origin_y = static_cast<int>(floorf(source_y));
      for (int offset_y = -1; offset_y <= 2; ++offset_y) {
        const int sample_y = max(0, min(image.height - 1, origin_y + offset_y));
        const float weight_y = cubic_weight(source_y - (origin_y + offset_y));
        const auto* row = image.pixels +
                          static_cast<std::size_t>(sample_y) * image.pitch;
        for (int offset_x = -1; offset_x <= 2; ++offset_x) {
          const int sample_x = max(0, min(image.width - 1, origin_x + offset_x));
          const float weight_x = cubic_weight(source_x - (origin_x + offset_x));
          value += row[sample_x * 3 + channel] * weight_x * weight_y;
        }
      }
      // cv2.resize keeps uint8 input in uint8. Quantize before normalization
      // so the fused GPU path has the same model input contract as v1.
      values[channel] = roundf(min(255.0F, max(0.0F, value))) * scale;
    } else {
      const float upper = row0[x0 * 3 + channel] * (1.0F - wx) +
                          row0[x1 * 3 + channel] * wx;
      const float lower = row1[x0 * 3 + channel] * (1.0F - wx) +
                          row1[x1 * 3 + channel] * wx;
      const float value = upper * (1.0F - wy) + lower * wy;
      values[channel] = roundf(min(255.0F, max(0.0F, value))) * scale;
    }
  }
  const float means[3] = {mean.x, mean.y, mean.z};
  const float inv_stds[3] = {inverse_std.x, inverse_std.y, inverse_std.z};
  const std::size_t base = static_cast<std::size_t>(image_index) * plane * 3 +
                           static_cast<std::size_t>(y) * out_w + x;
  for (int channel = 0; channel < 3; ++channel) {
    const int source_channel = swap_red_blue ? 2 - channel : channel;
    output[base + static_cast<std::size_t>(channel) * plane] =
        (values[source_channel] - means[channel]) * inv_stds[channel];
  }
}

__device__ float distance(float x0, float y0, float x1, float y1) {
  return hypotf(x1 - x0, y1 - y0);
}

__global__ void warp_quad_normalize_kernel(
    float* output, int count, int out_w, int out_h, float3 mean,
    float3 inverse_std, float scale, bool swap_red_blue, float pad_value) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int image_index = blockIdx.z;
  if (x >= out_w || y >= out_h || image_index >= count) return;
  const auto quad = kQuads[image_index];
  const float source_width =
      max(distance(quad.x0, quad.y0, quad.x1, quad.y1),
          distance(quad.x3, quad.y3, quad.x2, quad.y2));
  const float source_height =
      max(distance(quad.x0, quad.y0, quad.x3, quad.y3),
          distance(quad.x1, quad.y1, quad.x2, quad.y2));
  const int resized_width = min(
      out_w, max(1, static_cast<int>(ceilf(out_h * source_width /
                                           max(source_height, 1.0F)))));
  const bool padded = x >= resized_width;
  const float u = (static_cast<float>(x) + 0.5F) / resized_width;
  const float v = (static_cast<float>(y) + 0.5F) / out_h;

  // Exact unit-square to quadrilateral homography.  Degenerate quads fall
  // back to the affine form, and all sampling remains on the GPU.
  const float dx1 = quad.x1 - quad.x2;
  const float dx2 = quad.x3 - quad.x2;
  const float dy1 = quad.y1 - quad.y2;
  const float dy2 = quad.y3 - quad.y2;
  const float sx = quad.x0 - quad.x1 + quad.x2 - quad.x3;
  const float sy = quad.y0 - quad.y1 + quad.y2 - quad.y3;
  const float determinant = dx1 * dy2 - dx2 * dy1;
  const float g = fabsf(determinant) > 1.0e-6F
                      ? (sx * dy2 - dx2 * sy) / determinant
                      : 0.0F;
  const float h = fabsf(determinant) > 1.0e-6F
                      ? (dx1 * sy - sx * dy1) / determinant
                      : 0.0F;
  const float a = quad.x1 - quad.x0 + g * quad.x1;
  const float b = quad.x3 - quad.x0 + h * quad.x3;
  const float d = quad.y1 - quad.y0 + g * quad.y1;
  const float e = quad.y3 - quad.y0 + h * quad.y3;
  const float denominator = max(1.0e-6F, g * u + h * v + 1.0F);
  const float source_x = (a * u + b * v + quad.x0) / denominator;
  const float source_y = (d * u + e * v + quad.y0) / denominator;
  const int x0 = max(0, min(quad.image.width - 1,
                            static_cast<int>(floorf(source_x))));
  const int y0 = max(0, min(quad.image.height - 1,
                            static_cast<int>(floorf(source_y))));
  const int x1 = min(quad.image.width - 1, x0 + 1);
  const int y1 = min(quad.image.height - 1, y0 + 1);
  const float wx = source_x - floorf(source_x);
  const float wy = source_y - floorf(source_y);
  const auto* row0 = quad.image.pixels +
                     static_cast<std::size_t>(y0) * quad.image.pitch;
  const auto* row1 = quad.image.pixels +
                     static_cast<std::size_t>(y1) * quad.image.pitch;
  const float means[3] = {mean.x, mean.y, mean.z};
  const float inv_stds[3] = {inverse_std.x, inverse_std.y, inverse_std.z};
  const std::size_t plane = static_cast<std::size_t>(out_w) * out_h;
  const std::size_t base = static_cast<std::size_t>(image_index) * plane * 3 +
                           static_cast<std::size_t>(y) * out_w + x;
  for (int channel = 0; channel < 3; ++channel) {
    const int source_channel = swap_red_blue ? 2 - channel : channel;
    float value = pad_value;
    if (!padded) {
      const float upper = row0[x0 * 3 + source_channel] * (1.0F - wx) +
                          row0[x1 * 3 + source_channel] * wx;
      const float lower = row1[x0 * 3 + source_channel] * (1.0F - wx) +
                          row1[x1 * 3 + source_channel] * wx;
      value = upper * (1.0F - wy) + lower * wy;
    }
    output[base + static_cast<std::size_t>(channel) * plane] =
        (value * scale - means[channel]) * inv_stds[channel];
  }
}

struct Candidate {
  float score;
  float left;
  float top;
  float right;
  float bottom;
};

__device__ float overlap(const Candidate& first, const Candidate& second) {
  const float left = max(first.left, second.left);
  const float top = max(first.top, second.top);
  const float right = min(first.right, second.right);
  const float bottom = min(first.bottom, second.bottom);
  const float intersection = max(0.0F, right - left) * max(0.0F, bottom - top);
  const float first_area = max(0.0F, first.right - first.left) *
                           max(0.0F, first.bottom - first.top);
  const float second_area = max(0.0F, second.right - second.left) *
                            max(0.0F, second.bottom - second.top);
  const float divisor = first_area + second_area - intersection;
  return divisor > 0.0F ? intersection / divisor : 0.0F;
}

// One block owns an image. Candidate selection and NMS never leave the GPU.
__global__ void filter_nms_kernel(const float* boxes, const float* scores,
                                  int anchors, float score_threshold,
                                  float iou_threshold, int max_output,
                                  Box* output, int* output_counts) {
  constexpr int kCandidates = 512;
  __shared__ Candidate candidates[kCandidates];
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = 0;
  }
  __syncthreads();
  const int image = blockIdx.x;
  for (int anchor = threadIdx.x; anchor < anchors; anchor += blockDim.x) {
    const float score = scores[image * anchors + anchor];
    if (score < score_threshold) {
      continue;
    }
    const int slot = atomicAdd(&count, 1);
    if (slot < kCandidates) {
      const auto box_offset = (image * anchors + anchor) * 4;
      candidates[slot] = Candidate{score, boxes[box_offset], boxes[box_offset + 1],
                                   boxes[box_offset + 2], boxes[box_offset + 3]};
    }
  }
  __syncthreads();
  int usable = min(count, kCandidates);
  // Sorting at most 512 records per image is small relative to the detector;
  // a single thread avoids global scratch allocations and keeps ordering exact.
  if (threadIdx.x == 0) {
    // The fast parallel fill above covers the normal sparse case. If more than
    // 512 anchors pass the threshold, rescan all anchors and retain the exact
    // global top 512 instead of depending on anchor order.
    if (count > kCandidates) {
      usable = 0;
      for (int anchor = 0; anchor < anchors; ++anchor) {
        const float score = scores[image * anchors + anchor];
        if (score < score_threshold) continue;
        const auto box_offset = (image * anchors + anchor) * 4;
        const Candidate value{score, boxes[box_offset], boxes[box_offset + 1],
                              boxes[box_offset + 2], boxes[box_offset + 3]};
        if (usable < kCandidates) {
          candidates[usable++] = value;
        } else if (score <= candidates[kCandidates - 1].score) {
          continue;
        } else {
          candidates[kCandidates - 1] = value;
        }
        int position = usable - 1;
        while (position > 0 &&
               candidates[position - 1].score < candidates[position].score) {
          const Candidate swap = candidates[position - 1];
          candidates[position - 1] = candidates[position];
          candidates[position] = swap;
          --position;
        }
      }
    }
    for (int i = 1; i < usable; ++i) {
      const Candidate value = candidates[i];
      int position = i;
      while (position > 0 && candidates[position - 1].score < value.score) {
        candidates[position] = candidates[position - 1];
        --position;
      }
      candidates[position] = value;
    }
    int kept = 0;
    for (int i = 0; i < usable && kept < max_output; ++i) {
      bool suppressed = false;
      for (int previous = 0; previous < kept; ++previous) {
        const auto& accepted = output[image * max_output + previous];
        const Candidate accepted_candidate{accepted.score, accepted.left,
                                           accepted.top, accepted.right,
                                           accepted.bottom};
        if (overlap(candidates[i], accepted_candidate) > iou_threshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        const auto& value = candidates[i];
        output[image * max_output + kept] =
            Box{value.left, value.top, value.right, value.bottom, value.score, image};
        ++kept;
      }
    }
    output_counts[image] = kept;
  }
}

__global__ void ctc_argmax_kernel(const float* probabilities, int steps,
                                  int classes, int max_tokens, int* tokens,
                                  int* token_counts) {
  const int image = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }
  int previous = 0;
  int count = 0;
  for (int step = 0; step < steps; ++step) {
    const float* row = probabilities +
                       (static_cast<std::size_t>(image) * steps + step) * classes;
    int best = 0;
    float best_score = row[0];
    for (int token = 1; token < classes; ++token) {
      if (row[token] > best_score) {
        best = token;
        best_score = row[token];
      }
    }
    if (best != 0 && best != previous && count < max_tokens) {
      tokens[image * max_tokens + count++] = best;
    }
    previous = best;
  }
  token_counts[image] = count;
}

__global__ void mask_best_kernel(const float* predictions, int anchors,
                                 int values_per_anchor, float threshold,
                                 int* labels) {
  extern __shared__ unsigned char storage[];
  auto* scores = reinterpret_cast<float*>(storage);
  auto* classes = reinterpret_cast<int*>(scores + blockDim.x);
  const int image = blockIdx.x;
  float best_score = threshold;
  int best_class = 0;
  for (int anchor = threadIdx.x; anchor < anchors; anchor += blockDim.x) {
    const float* row = predictions +
                       (static_cast<std::size_t>(image) * anchors + anchor) *
                           values_per_anchor;
    const float class0 = row[4] * row[5];
    const float class1 = row[4] * row[6];
    const float candidate = max(class0, class1);
    if (candidate > best_score) {
      best_score = candidate;
      best_class = class1 > class0 ? 1 : 0;
    }
  }
  scores[threadIdx.x] = best_score;
  classes[threadIdx.x] = best_class;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride && scores[threadIdx.x + stride] > scores[threadIdx.x]) {
      scores[threadIdx.x] = scores[threadIdx.x + stride];
      classes[threadIdx.x] = classes[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    labels[image] = scores[0] >= threshold ? classes[0] : 0;
  }
}

__device__ int component_root(int* parents, int node) {
  int root = node;
  while (parents[root] != root) root = parents[root];
  while (parents[node] != node) {
    const int next = parents[node];
    parents[node] = root;
    node = next;
  }
  return root;
}

__device__ float component_cell_score(const float* map, int height, int width,
                                      int grid_x, int grid_y) {
  constexpr int stride = 2;
  float score = 0.0F;
  for (int dy = 0; dy < stride; ++dy) {
    const int y = grid_y * stride + dy;
    if (y >= height) continue;
    for (int dx = 0; dx < stride; ++dx) {
      const int x = grid_x * stride + dx;
      if (x < width) score = max(score, map[y * width + x]);
    }
  }
  return score;
}

__global__ void plate_quads_kernel(
    const float* maps, int height, int width, float pixel_threshold,
    float box_threshold, int* all_parents, int* all_counts,
    float* all_scores, PlateQuad* quads) {
  if (threadIdx.x != 0) return;
  constexpr int stride = 2;
  const int image = blockIdx.x;
  const int grid_width = (width + stride - 1) / stride;
  const int grid_height = (height + stride - 1) / stride;
  const int cells = grid_width * grid_height;
  const float* map = maps + static_cast<std::size_t>(image) * height * width;
  int* parents = all_parents + static_cast<std::size_t>(image) * cells;
  int* counts = all_counts + static_cast<std::size_t>(image) * cells;
  float* scores = all_scores + static_cast<std::size_t>(image) * cells;
  for (int cell = 0; cell < cells; ++cell) {
    parents[cell] = -1;
    counts[cell] = 0;
    scores[cell] = 0.0F;
  }
  for (int gy = 0; gy < grid_height; ++gy) {
    for (int gx = 0; gx < grid_width; ++gx) {
      const float cell_score =
          component_cell_score(map, height, width, gx, gy);
      if (cell_score <= pixel_threshold) continue;
      const int cell = gy * grid_width + gx;
      parents[cell] = cell;
      const int neighbors[4] = {
          gx > 0 ? cell - 1 : -1,
          gy > 0 ? cell - grid_width : -1,
          gx > 0 && gy > 0 ? cell - grid_width - 1 : -1,
          gx + 1 < grid_width && gy > 0 ? cell - grid_width + 1 : -1};
      for (const int neighbor : neighbors) {
        if (neighbor < 0 || parents[neighbor] < 0) continue;
        const int cell_root = component_root(parents, cell);
        const int neighbor_root = component_root(parents, neighbor);
        if (cell_root == neighbor_root) continue;
        const int root = min(cell_root, neighbor_root);
        parents[max(cell_root, neighbor_root)] = root;
        parents[cell] = root;
      }
    }
  }
  for (int cell = 0; cell < cells; ++cell) {
    if (parents[cell] < 0) continue;
    const int root = component_root(parents, cell);
    parents[cell] = root;
    ++counts[root];
    const int gy = cell / grid_width;
    const int gx = cell - gy * grid_width;
    scores[root] += component_cell_score(map, height, width, gx, gy);
  }
  int best_root = -1;
  int best_count = 0;
  float best_score = 0.0F;
  for (int cell = 0; cell < cells; ++cell) {
    if (counts[cell] < 3) continue;
    const float score = scores[cell] / counts[cell];
    if (score >= box_threshold &&
        (counts[cell] > best_count ||
         (counts[cell] == best_count && score > best_score))) {
      best_root = cell;
      best_count = counts[cell];
      best_score = score;
    }
  }
  if (best_root < 0) {
    quads[image] = PlateQuad{0, 0, 0, 0, 0, 0, 0, 0, best_score, 0};
    return;
  }
  float sum_x = 0.0F;
  float sum_y = 0.0F;
  float sum_xx = 0.0F;
  float sum_yy = 0.0F;
  float sum_xy = 0.0F;
  for (int cell = 0; cell < cells; ++cell) {
    if (parents[cell] != best_root) continue;
    const int gy = cell / grid_width;
    const int gx = cell - gy * grid_width;
    const float x = min(static_cast<float>(width - 1),
                        gx * stride + stride * 0.5F);
    const float y = min(static_cast<float>(height - 1),
                        gy * stride + stride * 0.5F);
    sum_x += x;
    sum_y += y;
    sum_xx += x * x;
    sum_yy += y * y;
    sum_xy += x * y;
  }
  const float center_x = sum_x / best_count;
  const float center_y = sum_y / best_count;
  const float covariance_xx = sum_xx / best_count - center_x * center_x;
  const float covariance_yy = sum_yy / best_count - center_y * center_y;
  const float covariance_xy = sum_xy / best_count - center_x * center_y;
  const float angle =
      0.5F * atan2f(2.0F * covariance_xy, covariance_xx - covariance_yy);
  const float ux = cosf(angle);
  const float uy = sinf(angle);
  const float vx = -uy;
  const float vy = ux;
  float min_u = 1.0e20F;
  float max_u = -1.0e20F;
  float min_v = 1.0e20F;
  float max_v = -1.0e20F;
  for (int cell = 0; cell < cells; ++cell) {
    if (parents[cell] != best_root) continue;
    const int gy = cell / grid_width;
    const int gx = cell - gy * grid_width;
    const float x = gx * stride + stride * 0.5F - center_x;
    const float y = gy * stride + stride * 0.5F - center_y;
    const float projection_u = x * ux + y * uy;
    const float projection_v = x * vx + y * vy;
    min_u = min(min_u, projection_u);
    max_u = max(max_u, projection_u);
    min_v = min(min_v, projection_v);
    max_v = max(max_v, projection_v);
  }
  const float box_width = max_u - min_u + stride;
  const float box_height = max_v - min_v + stride;
  if (box_width < 3.0F || box_height < 3.0F) {
    quads[image] = PlateQuad{0, 0, 0, 0, 0, 0, 0, 0, best_score, 0};
    return;
  }
  const float unclip = box_width * box_height * 1.5F /
                       max(1.0F, 2.0F * (box_width + box_height));
  min_u -= unclip;
  max_u += unclip;
  min_v -= unclip;
  max_v += unclip;
  const auto point_x = [&](float u, float v) {
    return min(static_cast<float>(width - 1),
               max(0.0F, center_x + u * ux + v * vx));
  };
  const auto point_y = [&](float u, float v) {
    return min(static_cast<float>(height - 1),
               max(0.0F, center_y + u * uy + v * vy));
  };
  quads[image] = PlateQuad{
      point_x(min_u, min_v), point_y(min_u, min_v),
      point_x(max_u, min_v), point_y(max_u, min_v),
      point_x(max_u, max_v), point_y(max_u, max_v),
      point_x(min_u, max_v), point_y(min_u, max_v), best_score, 1};
}

}  // namespace

void launch_resize_normalize(const ImageView* host_images, int count,
                             float* output, int output_width,
                             int output_height, const float mean[3],
                             const float inverse_std[3], bool divide_255,
                             bool swap_red_blue, ResizeMode mode,
                             Interpolation interpolation, float pad_value,
                             cudaStream_t stream) {
  if (count < 1 || count > 128) {
    throw std::invalid_argument("resize batch must be between 1 and 128");
  }
  check_cuda(cudaMemcpyToSymbolAsync(kImages, host_images,
                                     sizeof(ImageView) * count, 0,
                                     cudaMemcpyHostToDevice, stream),
             "cudaMemcpyToSymbolAsync image views");
  const dim3 block(16, 16);
  const dim3 grid((output_width + block.x - 1) / block.x,
                  (output_height + block.y - 1) / block.y, count);
  resize_normalize_kernel<<<grid, block, 0, stream>>>(
      output, count, output_width, output_height,
      make_float3(mean[0], mean[1], mean[2]),
      make_float3(inverse_std[0], inverse_std[1], inverse_std[2]),
      divide_255 ? 1.0F / 255.0F : 1.0F, swap_red_blue, mode,
      interpolation, pad_value);
  check_cuda(cudaGetLastError(), "resize_normalize_kernel");
}

void launch_warp_quad_normalize(
    const QuadView* host_quads, int count, float* output, int output_width,
    int output_height, const float mean[3], const float inverse_std[3],
    bool divide_255, bool swap_red_blue, float pad_value,
    cudaStream_t stream) {
  if (count < 1 || count > 128) {
    throw std::invalid_argument("quad warp batch must be between 1 and 128");
  }
  check_cuda(cudaMemcpyToSymbolAsync(kQuads, host_quads,
                                     sizeof(QuadView) * count, 0,
                                     cudaMemcpyHostToDevice, stream),
             "cudaMemcpyToSymbolAsync quad views");
  const dim3 block(16, 16);
  const dim3 grid((output_width + block.x - 1) / block.x,
                  (output_height + block.y - 1) / block.y, count);
  warp_quad_normalize_kernel<<<grid, block, 0, stream>>>(
      output, count, output_width, output_height,
      make_float3(mean[0], mean[1], mean[2]),
      make_float3(inverse_std[0], inverse_std[1], inverse_std[2]),
      divide_255 ? 1.0F / 255.0F : 1.0F, swap_red_blue, pad_value);
  check_cuda(cudaGetLastError(), "warp_quad_normalize_kernel");
}

void launch_mask_best(const float* predictions, int batch, int anchors,
                      int values_per_anchor, float threshold, int* labels,
                      cudaStream_t stream) {
  constexpr int threads = 256;
  mask_best_kernel<<<batch, threads, threads * (sizeof(float) + sizeof(int)), stream>>>(
      predictions, anchors, values_per_anchor, threshold, labels);
  check_cuda(cudaGetLastError(), "mask_best_kernel");
}

void launch_plate_quads(
    const float* probability_maps, int batch, int height, int width,
    float pixel_threshold, float box_threshold, int* component_parents,
    int* component_counts, float* component_scores, PlateQuad* quads,
    cudaStream_t stream) {
  plate_quads_kernel<<<batch, 32, 0, stream>>>(
      probability_maps, height, width, pixel_threshold, box_threshold,
      component_parents, component_counts, component_scores, quads);
  check_cuda(cudaGetLastError(), "plate_quads_kernel");
}

void launch_filter_nms(const float* boxes, const float* scores, int batch,
                       int anchors, float score_threshold,
                       float iou_threshold, int max_output, Box* output,
                       int* output_counts, cudaStream_t stream) {
  filter_nms_kernel<<<batch, 256, 0, stream>>>(
      boxes, scores, anchors, score_threshold, iou_threshold, max_output,
      output, output_counts);
  check_cuda(cudaGetLastError(), "filter_nms_kernel");
}

void launch_ctc_argmax(const float* probabilities, int batch, int steps,
                       int classes, int max_tokens, int* tokens,
                       int* token_counts, cudaStream_t stream) {
  ctc_argmax_kernel<<<batch, 32, 0, stream>>>(probabilities, steps, classes,
                                              max_tokens, tokens, token_counts);
  check_cuda(cudaGetLastError(), "ctc_argmax_kernel");
}

}  // namespace pvr
