#pragma once

#include "pvr/image_decoder.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace pvr {

struct Box {
  float left;
  float top;
  float right;
  float bottom;
  float score;
  int image_index;
};

struct ImageView {
  const std::uint8_t* pixels;
  int width;
  int height;
  std::size_t pitch;
};

struct PlateQuad {
  float x0;
  float y0;
  float x1;
  float y1;
  float x2;
  float y2;
  float x3;
  float y3;
  float score;
  int found;
};

struct QuadView {
  ImageView image;
  float x0;
  float y0;
  float x1;
  float y1;
  float x2;
  float y2;
  float x3;
  float y3;
};

enum class ResizeMode : int {
  stretch = 0,
  letterbox_center = 1,
  fit_height_left = 2,
};

enum class Interpolation : int { linear = 0, cubic = 1 };

void launch_resize_normalize(const ImageView* host_images, int count,
                             float* output, int output_width,
                             int output_height, const float mean[3],
                             const float inverse_std[3], bool divide_255,
                             bool swap_red_blue, ResizeMode mode,
                             Interpolation interpolation, float pad_value,
                             cudaStream_t stream);

void launch_warp_quad_normalize(const QuadView* host_quads, int count,
                                float* output, int output_width,
                                int output_height, const float mean[3],
                                const float inverse_std[3], bool divide_255,
                                bool swap_red_blue, float pad_value,
                                cudaStream_t stream);

void launch_filter_nms(const float* boxes, const float* scores, int batch,
                       int anchors, float score_threshold,
                       float iou_threshold, int max_output, Box* output,
                       int* output_counts, cudaStream_t stream);

void launch_ctc_argmax(const float* probabilities, int batch, int steps,
                       int classes, int max_tokens, int* tokens,
                       int* token_counts, cudaStream_t stream);

void launch_mask_best(const float* predictions, int batch, int anchors,
                      int values_per_anchor, float threshold, int* labels,
                      cudaStream_t stream);

void launch_plate_quads(const float* probability_maps, int batch, int height,
                        int width, float pixel_threshold,
                        float box_threshold, int* component_parents,
                        int* component_counts, float* component_scores,
                        PlateQuad* quads, cudaStream_t stream);

}  // namespace pvr
