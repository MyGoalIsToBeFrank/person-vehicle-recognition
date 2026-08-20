#include "pvr/pipeline.hpp"

#include "pvr/cuda_utils.hpp"
#include "pvr/image_decoder.hpp"
#include "pvr/kernels.hpp"
#include "pvr/tensorrt_model.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pvr {
namespace {

constexpr int kDetectorSide = 640;
constexpr int kMaxDetections = 100;
constexpr int kPlateSide = 736;
constexpr int kPlateRecWidth = 320;
constexpr int kPlateRecHeight = 48;
constexpr std::size_t kRoiChunk = 128;

ShapeRange fixed_image_profile(int min_batch, int opt_batch, int max_batch,
                               int height, int width) {
  return {{min_batch, 3, height, width},
          {opt_batch, 3, height, width},
          {max_batch, 3, height, width}};
}

std::unordered_map<std::string, ShapeRange> detector_profiles() {
  return {{"image", fixed_image_profile(1, 32, 64, 640, 640)},
          {"scale_factor", {{1, 2}, {32, 2}, {64, 2}}}};
}

std::unordered_map<std::string, ShapeRange> single_image_profile(
    const char* name, int max_batch, int height, int width) {
  return {{name, fixed_image_profile(1, std::min(32, max_batch), max_batch,
                                     height, width)}};
}

const DeviceTensor& output_with_last_dimension(
    const std::vector<DeviceTensor>& tensors, std::int64_t dimension) {
  const auto found = std::find_if(tensors.begin(), tensors.end(),
                                  [dimension](const DeviceTensor& tensor) {
                                    return !tensor.shape.empty() &&
                                           tensor.shape.back() == dimension;
                                  });
  if (found == tensors.end()) {
    throw std::runtime_error("expected TensorRT output dimension not found");
  }
  if (found->type != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error("pipeline requires FP32 model output tensors");
  }
  return *found;
}

const DeviceTensor& only_output(const std::vector<DeviceTensor>& tensors) {
  if (tensors.size() != 1 || tensors[0].type != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error("model output contract mismatch");
  }
  return tensors[0];
}

std::optional<ImageView> expanded_crop_view(const DeviceImage& image,
                                            const Box& box, float scale) {
  // Preserve the v1/OpenCV crop contract exactly: truncate the detector box
  // before expansion and treat right/bottom as exclusive slice bounds.
  int left = static_cast<int>(box.left);
  int top = static_cast<int>(box.top);
  int right = static_cast<int>(box.right);
  int bottom = static_cast<int>(box.bottom);
  const float center_x = (left + right) * 0.5F;
  const float center_y = (top + bottom) * 0.5F;
  float half_height = (bottom - top) * scale * 0.5F;
  float half_width = (right - left) * scale * 0.5F;
  if (half_height > half_width * 4.0F / 3.0F) {
    half_width = half_height * 0.75F;
  }
  left = std::max(0, static_cast<int>(center_x - half_width));
  right = std::min(image.width - 1,
                   static_cast<int>(center_x + half_width));
  top = std::max(0, static_cast<int>(center_y - half_height));
  bottom = std::min(image.height - 1,
                    static_cast<int>(center_y + half_height));
  if (right <= left || bottom <= top) return std::nullopt;
  const auto* origin = static_cast<const std::uint8_t*>(image.pixels.data());
  return ImageView{origin + static_cast<std::size_t>(top) * image.pitch + left * 3,
                   right - left, bottom - top, image.pitch};
}

std::optional<ImageView> upper_crop_view(const DeviceImage& image,
                                         const Box& box, float ratio) {
  const int left = std::clamp(static_cast<int>(box.left), 0, image.width);
  const int top = std::clamp(static_cast<int>(box.top), 0, image.height);
  const int right = std::clamp(static_cast<int>(box.right), 0, image.width);
  const int original_bottom =
      std::clamp(static_cast<int>(box.bottom), 0, image.height);
  const int bottom = std::min(
      original_bottom,
      top + std::max(1, static_cast<int>((original_bottom - top) * ratio)));
  if (right <= left || bottom <= top) return std::nullopt;
  const auto* origin = static_cast<const std::uint8_t*>(image.pixels.data());
  return ImageView{origin + static_cast<std::size_t>(top) * image.pitch + left * 3,
                   right - left, bottom - top, image.pitch};
}

float contained_overlap(const Box& first, const Box& second) {
  const float left = std::max(first.left, second.left);
  const float top = std::max(first.top, second.top);
  const float right = std::min(first.right, second.right);
  const float bottom = std::min(first.bottom, second.bottom);
  const float intersection = std::max(0.0F, right - left) *
                             std::max(0.0F, bottom - top);
  const float first_area = std::max(0.0F, first.right - first.left) *
                           std::max(0.0F, first.bottom - first.top);
  const float second_area = std::max(0.0F, second.right - second.left) *
                            std::max(0.0F, second.bottom - second.top);
  const float smaller = std::min(first_area, second_area);
  return smaller > 0.0F ? intersection / smaller : 0.0F;
}

std::string json_string(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20) {
          output << "\\u00" << std::hex << static_cast<int>(character);
        } else {
          output << character;
        }
    }
  }
  output << '"';
  return output.str();
}

std::string decode_person(const float* scores, int mask_label) {
  static const std::array<const char*, 3> ages = {"未满18岁", "18至60岁",
                                                   "60岁以上"};
  static const std::array<const char*, 3> directions = {"正面", "侧面", "背面"};
  static const std::array<const char*, 3> bags = {"手提包", "单肩包", "双肩包"};
  static const std::array<const char*, 4> upper = {"条纹", "标志", "格纹", "拼接"};
  static const std::array<const char*, 6> lower = {"条纹", "图案", "长外套",
                                                   "长裤", "短裤", "裙装"};
  const auto argmax = [scores](int begin, int end) {
    int best = begin;
    for (int index = begin + 1; index < end; ++index) {
      if (scores[index] > scores[best]) {
        best = index;
      }
    }
    return best;
  };
  const int bag = argmax(15, 18);
  const int upper_style = argmax(4, 8);
  std::vector<const char*> lower_styles;
  for (int index = 8; index < 14; ++index) {
    if (scores[index] > 0.5F) {
      lower_styles.push_back(lower[static_cast<std::size_t>(index - 8)]);
    }
  }
  if (lower_styles.empty()) {
    lower_styles.push_back(lower[static_cast<std::size_t>(argmax(8, 14) - 8)]);
  }
  std::ostringstream output;
  output << "{\"性别\":\"" << (scores[22] > 0.5F ? "女" : "男")
         << "\",\"年龄\":\"" << ages[static_cast<std::size_t>(argmax(19, 22) - 19)]
         << "\",\"朝向\":\""
         << directions[static_cast<std::size_t>(argmax(23, 26) - 23)]
         << "\",\"佩戴眼镜\":\"" << (scores[1] > 0.3F ? "是" : "否")
         << "\",\"佩戴帽子\":\"" << (scores[0] > 0.5F ? "是" : "否")
         << "\",\"手持物品\":\"" << (scores[18] > 0.6F ? "是" : "否")
         << "\",\"包\":\""
         << (scores[bag] > 0.5F ? bags[static_cast<std::size_t>(bag - 15)] : "无")
         << "\",\"上装\":{\"袖长\":\""
         << (scores[3] > scores[2] ? "长袖" : "短袖") << "\",\"款式\":[";
  if (scores[upper_style] > 0.5F) {
    output << '"' << upper[static_cast<std::size_t>(upper_style - 4)] << '"';
  }
  output << "]},\"下装\":[";
  for (std::size_t i = 0; i < lower_styles.size(); ++i) {
    if (i) output << ',';
    output << '"' << lower_styles[i] << '"';
  }
  output << "],\"鞋靴\":\"" << (scores[14] > 0.5F ? "靴子" : "非靴子")
         << "\",\"口罩\":\"" << (mask_label == 1 ? "佩戴口罩" : "未佩戴口罩")
         << "\"}";
  return output.str();
}

std::string decode_vehicle(const float* scores, const std::string& plate) {
  static const std::array<const char*, 10> colors = {
      "黄色", "橙色", "绿色", "灰色", "红色",
      "蓝色", "白色", "金色", "棕色", "黑色"};
  static const std::array<const char*, 9> types = {
      "轿车", "SUV", "厢式货车", "掀背车", "MPV",
      "皮卡", "公交车", "卡车", "旅行车"};
  int color = 0;
  int type = 10;
  for (int i = 1; i < 10; ++i) {
    if (scores[i] > scores[color]) color = i;
  }
  for (int i = 11; i < 19; ++i) {
    if (scores[i] > scores[type]) type = i;
  }
  std::ostringstream output;
  output << "{\"颜色\":\""
         << (scores[color] >= 0.5F ? colors[static_cast<std::size_t>(color)] : "未知")
         << "\",\"车型\":\""
         << (scores[type] >= 0.5F ? types[static_cast<std::size_t>(type - 10)] : "未知")
         << "\",\"车牌\":" << json_string(plate) << '}';
  return output.str();
}

struct RoiJob {
  int image = 0;
  Box detection{};
  ImageView body{};
  std::optional<ImageView> head;
};

}  // namespace

class Pipeline::Impl {
 public:
  explicit Impl(const Config& config)
      : decoder_(config.max_image_pixels) {
    const auto model_dir = std::filesystem::path(config.model_dir);
    const auto person_attr_profile = single_image_profile("x", 128, 256, 192);
    const auto vehicle_attr_profile = single_image_profile("x", 128, 192, 256);
    const auto mask_profile = single_image_profile("images", 128, 640, 640);
    const auto plate_det_profile =
        single_image_profile("x", 64, kPlateSide, kPlateSide);
    const auto plate_rec_profile =
        single_image_profile("x", 128, kPlateRecHeight, kPlateRecWidth);
    const auto prepare = [&](const char* file,
                             const std::unordered_map<std::string, ShapeRange>& profile) {
      TensorRtModel temporary(model_dir / file, config.engine_cache_dir, profile,
                              "fp16");
    };
    // Build every cache entry while no other execution context is resident.
    // Otherwise later models lose high-memory tactics merely because earlier
    // contexts consumed VRAM during first start.
    prepare("person_detector.onnx", detector_profiles());
    prepare("vehicle_detector.onnx", detector_profiles());
    prepare("person_attribute.onnx", person_attr_profile);
    prepare("vehicle_attribute.onnx", vehicle_attr_profile);
    prepare("face_mask.onnx", mask_profile);
    prepare("plate_det.onnx", plate_det_profile);
    prepare("plate_rec.onnx", plate_rec_profile);
    person_detector_ = std::make_unique<TensorRtModel>(
        model_dir / "person_detector.onnx", config.engine_cache_dir,
        detector_profiles(), "fp16");
    vehicle_detector_ = std::make_unique<TensorRtModel>(
        model_dir / "vehicle_detector.onnx", config.engine_cache_dir,
        detector_profiles(), "fp16");
    person_attribute_ = std::make_unique<TensorRtModel>(
        model_dir / "person_attribute.onnx", config.engine_cache_dir,
        person_attr_profile, "fp16");
    vehicle_attribute_ = std::make_unique<TensorRtModel>(
        model_dir / "vehicle_attribute.onnx", config.engine_cache_dir,
        vehicle_attr_profile, "fp16");
    face_mask_ = std::make_unique<TensorRtModel>(
        model_dir / "face_mask.onnx", config.engine_cache_dir, mask_profile,
        "fp16");
    plate_detector_ = std::make_unique<TensorRtModel>(
        model_dir / "plate_det.onnx", config.engine_cache_dir,
        plate_det_profile, "fp16");
    plate_recognizer_ = std::make_unique<TensorRtModel>(
        model_dir / "plate_rec.onnx", config.engine_cache_dir,
        plate_rec_profile, "fp16");
    person_context_memory_.resize(person_detector_->context_memory_bytes());
    vehicle_context_memory_.resize(vehicle_detector_->context_memory_bytes());
    const std::size_t sequential_bytes = std::max(
        {person_attribute_->context_memory_bytes(),
         vehicle_attribute_->context_memory_bytes(),
         face_mask_->context_memory_bytes(),
         plate_detector_->context_memory_bytes(),
         plate_recognizer_->context_memory_bytes()});
    sequential_context_memory_.resize(sequential_bytes);
    person_detector_->bind_context_memory(person_context_memory_.data(),
                                          person_context_memory_.capacity());
    vehicle_detector_->bind_context_memory(vehicle_context_memory_.data(),
                                           vehicle_context_memory_.capacity());
    for (auto* model : {person_attribute_.get(), vehicle_attribute_.get(),
                        face_mask_.get(), plate_detector_.get(),
                        plate_recognizer_.get()}) {
      model->bind_context_memory(sequential_context_memory_.data(),
                                 sequential_context_memory_.capacity());
    }
    context_pool_bytes_ = person_context_memory_.capacity() +
                          vehicle_context_memory_.capacity() +
                          sequential_context_memory_.capacity();
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");
    gpu_name_ = properties.name;
    std::ifstream dictionary(std::filesystem::path(config.model_dir) /
                             "rec_word_dict.txt");
    if (!dictionary) {
      throw std::runtime_error("missing PP-OCRv3 recognition dictionary");
    }
    characters_.push_back("blank");
    for (std::string line; std::getline(dictionary, line);) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (!line.empty()) characters_.push_back(std::move(line));
    }
    characters_.push_back(" ");
  }

  std::vector<PipelineResult> run(const std::vector<Task>& tasks);

  std::unordered_map<std::string, std::string> health() const {
    return {{"gpu", gpu_name_},
            {"engine_precision", "fp16"},
            {"person_detector_cache", person_detector_->cache_key()},
            {"vehicle_detector_cache", vehicle_detector_->cache_key()},
            {"person_attribute_cache", person_attribute_->cache_key()},
            {"vehicle_attribute_cache", vehicle_attribute_->cache_key()},
            {"face_mask_cache", face_mask_->cache_key()},
            {"plate_detector_cache", plate_detector_->cache_key()},
            {"plate_recognizer_cache", plate_recognizer_->cache_key()},
            {"trt_context_memory_pool_bytes", std::to_string(context_pool_bytes_)},
            {"trt_context_memory_layout", "two detector pools + one shared sequential pool"},
            {"cuda_graphs", "fixed batches 1/8/16/32/64/128 with safe fallback"},
            {"ocr", "PaddleDetection PP-OCRv3 det+rec"}};
  }

 private:
  void enqueue_detect(TensorRtModel& model, const DeviceTensor& image,
                      const DeviceTensor& scale, int batch,
                      DeviceBuffer& nms_boxes, DeviceBuffer& nms_counts,
                      cudaStream_t stream);
  std::vector<Box> collect_detections(int batch, DeviceBuffer& nms_boxes,
                                      DeviceBuffer& nms_counts,
                                      cudaStream_t stream);
  std::vector<float> attributes(TensorRtModel& model,
                                const std::vector<ImageView>& views, int width,
                                int height, int values);
  std::vector<int> masks(const std::vector<ImageView>& views);
  std::vector<std::string> plates(const std::vector<ImageView>& vehicles);

  ImageDecoder decoder_;
  DeviceBuffer person_context_memory_;
  DeviceBuffer vehicle_context_memory_;
  DeviceBuffer sequential_context_memory_;
  std::unique_ptr<TensorRtModel> person_detector_;
  std::unique_ptr<TensorRtModel> vehicle_detector_;
  std::unique_ptr<TensorRtModel> person_attribute_;
  std::unique_ptr<TensorRtModel> vehicle_attribute_;
  std::unique_ptr<TensorRtModel> face_mask_;
  std::unique_ptr<TensorRtModel> plate_detector_;
  std::unique_ptr<TensorRtModel> plate_recognizer_;
  CudaStream primary_stream_;
  CudaStream secondary_stream_;
  DeviceBuffer detector_input_;
  DeviceBuffer scale_input_;
  DeviceBuffer person_nms_boxes_;
  DeviceBuffer person_nms_counts_;
  DeviceBuffer vehicle_nms_boxes_;
  DeviceBuffer vehicle_nms_counts_;
  DeviceBuffer roi_input_;
  DeviceBuffer labels_device_;
  DeviceBuffer plate_quads_device_;
  DeviceBuffer plate_component_parents_;
  DeviceBuffer plate_component_counts_;
  DeviceBuffer plate_component_scores_;
  DeviceBuffer tokens_device_;
  DeviceBuffer token_counts_device_;
  std::vector<DeviceImage> decoded_pool_;
  std::vector<std::string> characters_;
  std::size_t context_pool_bytes_ = 0;
  std::string gpu_name_ = "CUDA";
};

void Pipeline::Impl::enqueue_detect(
    TensorRtModel& model, const DeviceTensor& image, const DeviceTensor& scale,
    int batch, DeviceBuffer& nms_boxes, DeviceBuffer& nms_counts,
    cudaStream_t stream) {
  auto outputs = model.infer({image, scale}, stream);
  const auto& boxes = output_with_last_dimension(outputs, 4);
  const auto score_found = std::find_if(
      outputs.begin(), outputs.end(), [&boxes](const DeviceTensor& tensor) {
        return tensor.data != boxes.data;
      });
  if (score_found == outputs.end()) {
    throw std::runtime_error("detector score output not found");
  }
  const auto& scores = *score_found;
  if (scores.type != nvinfer1::DataType::kFLOAT || boxes.shape.size() != 3) {
    throw std::runtime_error("detector raw output contract mismatch");
  }
  const int anchors = static_cast<int>(boxes.shape[1]);
  nms_boxes.resize(static_cast<std::size_t>(batch) * kMaxDetections * sizeof(Box));
  nms_counts.resize(static_cast<std::size_t>(batch) * sizeof(int));
  launch_filter_nms(static_cast<const float*>(boxes.data),
                    static_cast<const float*>(scores.data), batch, anchors, 0.5F,
                    0.7F, kMaxDetections, static_cast<Box*>(nms_boxes.data()),
                    static_cast<int*>(nms_counts.data()), stream);
}

std::vector<Box> Pipeline::Impl::collect_detections(
    int batch, DeviceBuffer& nms_boxes, DeviceBuffer& nms_counts,
    cudaStream_t stream) {
  std::vector<int> counts(batch);
  std::vector<Box> all(static_cast<std::size_t>(batch) * kMaxDetections);
  check_cuda(cudaMemcpyAsync(counts.data(), nms_counts.data(),
                             counts.size() * sizeof(int), cudaMemcpyDeviceToHost,
                             stream),
             "copy detector counts");
  check_cuda(cudaMemcpyAsync(all.data(), nms_boxes.data(),
                             all.size() * sizeof(Box), cudaMemcpyDeviceToHost,
                             stream),
             "copy detector boxes");
  check_cuda(cudaStreamSynchronize(stream), "synchronize detector");
  std::vector<Box> compact;
  for (int image_index = 0; image_index < batch; ++image_index) {
    const int count = std::clamp(counts[image_index], 0, kMaxDetections);
    for (int i = 0; i < count; ++i) {
      compact.push_back(all[static_cast<std::size_t>(image_index) * kMaxDetections + i]);
    }
  }
  return compact;
}

std::vector<float> Pipeline::Impl::attributes(
    TensorRtModel& model, const std::vector<ImageView>& views, int width,
    int height, int values) {
  std::vector<float> result(views.size() * static_cast<std::size_t>(values));
  static constexpr float mean[3] = {0.485F, 0.456F, 0.406F};
  static constexpr float inverse_std[3] = {1.0F / 0.229F, 1.0F / 0.224F,
                                            1.0F / 0.225F};
  for (std::size_t offset = 0; offset < views.size(); offset += kRoiChunk) {
    const int count = static_cast<int>(std::min(kRoiChunk, views.size() - offset));
    roi_input_.resize(static_cast<std::size_t>(count) * 3 * height * width * sizeof(float));
    launch_resize_normalize(views.data() + offset, count,
                            static_cast<float*>(roi_input_.data()), width, height,
                            mean, inverse_std, true, false, ResizeMode::stretch,
                            Interpolation::linear, 0.0F, primary_stream_);
    auto outputs = model.infer(
        {{"x", roi_input_.data(), {count, 3, height, width},
          nvinfer1::DataType::kFLOAT, roi_input_.capacity()}},
        primary_stream_);
    const auto& tensor = only_output(outputs);
    check_cuda(cudaMemcpyAsync(result.data() + offset * values, tensor.data,
                               static_cast<std::size_t>(count) * values * sizeof(float),
                               cudaMemcpyDeviceToHost, primary_stream_),
               "copy attribute scores");
    check_cuda(cudaStreamSynchronize(primary_stream_), "synchronize attributes");
  }
  return result;
}

std::vector<int> Pipeline::Impl::masks(const std::vector<ImageView>& views) {
  std::vector<int> result(views.size(), 0);
  static constexpr float zeros[3] = {0.0F, 0.0F, 0.0F};
  static constexpr float ones[3] = {1.0F, 1.0F, 1.0F};
  for (std::size_t offset = 0; offset < views.size(); offset += kRoiChunk) {
    const int count = static_cast<int>(std::min(kRoiChunk, views.size() - offset));
    roi_input_.resize(static_cast<std::size_t>(count) * 3 * 640 * 640 * sizeof(float));
    launch_resize_normalize(views.data() + offset, count,
                            static_cast<float*>(roi_input_.data()), 640, 640,
                            zeros, ones, true, false,
                            ResizeMode::letterbox_center, Interpolation::linear,
                            114.0F,
                            primary_stream_);
    auto mask_outputs = face_mask_->infer(
        {{"images", roi_input_.data(), {count, 3, 640, 640},
          nvinfer1::DataType::kFLOAT, roi_input_.capacity()}}, primary_stream_);
    const auto& tensor = only_output(mask_outputs);
    if (tensor.shape.size() != 3 || tensor.shape[2] < 7) {
      throw std::runtime_error("face mask output contract mismatch");
    }
    labels_device_.resize(static_cast<std::size_t>(count) * sizeof(int));
    launch_mask_best(static_cast<const float*>(tensor.data), count,
                     static_cast<int>(tensor.shape[1]),
                     static_cast<int>(tensor.shape[2]), 0.5F,
                     static_cast<int*>(labels_device_.data()), primary_stream_);
    check_cuda(cudaMemcpyAsync(result.data() + offset, labels_device_.data(),
                               static_cast<std::size_t>(count) * sizeof(int),
                               cudaMemcpyDeviceToHost, primary_stream_),
               "copy mask labels");
    check_cuda(cudaStreamSynchronize(primary_stream_), "synchronize mask");
  }
  return result;
}

std::vector<std::string> Pipeline::Impl::plates(
    const std::vector<ImageView>& vehicles) {
  std::vector<std::string> result(vehicles.size(), "未识别");
  if (vehicles.empty()) return result;
  static constexpr float det_mean[3] = {0.485F, 0.456F, 0.406F};
  static constexpr float det_inverse_std[3] = {1.0F / 0.229F,
                                                1.0F / 0.224F,
                                                1.0F / 0.225F};
  std::vector<QuadView> plate_views;
  std::vector<std::size_t> owners;
  for (std::size_t offset = 0; offset < vehicles.size(); offset += 64) {
    const int count = static_cast<int>(std::min<std::size_t>(64, vehicles.size() - offset));
    roi_input_.resize(static_cast<std::size_t>(count) * 3 * kPlateSide * kPlateSide *
                      sizeof(float));
    launch_resize_normalize(vehicles.data() + offset, count,
                            static_cast<float*>(roi_input_.data()), kPlateSide,
                            kPlateSide, det_mean, det_inverse_std, true, true,
                            ResizeMode::stretch, Interpolation::linear, 0.0F,
                            primary_stream_);
    auto detector_outputs = plate_detector_->infer(
        {{"x", roi_input_.data(), {count, 3, kPlateSide, kPlateSide},
          nvinfer1::DataType::kFLOAT, roi_input_.capacity()}}, primary_stream_);
    const auto& map = only_output(detector_outputs);
    if (map.shape.size() != 4) {
      throw std::runtime_error("PP-OCRv3 detector output contract mismatch");
    }
    const int map_h = static_cast<int>(map.shape[2]);
    const int map_w = static_cast<int>(map.shape[3]);
    const auto component_cells = static_cast<std::size_t>((map_h + 1) / 2) *
                                 static_cast<std::size_t>((map_w + 1) / 2);
    plate_component_parents_.resize(static_cast<std::size_t>(count) *
                                    component_cells * sizeof(int));
    plate_component_counts_.resize(static_cast<std::size_t>(count) *
                                   component_cells * sizeof(int));
    plate_component_scores_.resize(static_cast<std::size_t>(count) *
                                   component_cells * sizeof(float));
    plate_quads_device_.resize(static_cast<std::size_t>(count) *
                               sizeof(PlateQuad));
    launch_plate_quads(
        static_cast<const float*>(map.data), count, map_h, map_w, 0.3F, 0.6F,
        static_cast<int*>(plate_component_parents_.data()),
        static_cast<int*>(plate_component_counts_.data()),
        static_cast<float*>(plate_component_scores_.data()),
        static_cast<PlateQuad*>(plate_quads_device_.data()), primary_stream_);
    std::vector<PlateQuad> boxes(count);
    check_cuda(cudaMemcpyAsync(boxes.data(), plate_quads_device_.data(),
                               boxes.size() * sizeof(PlateQuad),
                               cudaMemcpyDeviceToHost, primary_stream_),
               "copy plate quads");
    check_cuda(cudaStreamSynchronize(primary_stream_), "synchronize plate detection");
    for (int i = 0; i < count; ++i) {
      if (!boxes[i].found) continue;
      const auto& vehicle = vehicles[offset + static_cast<std::size_t>(i)];
      const auto scale_x = static_cast<float>(vehicle.width) / map_w;
      const auto scale_y = static_cast<float>(vehicle.height) / map_h;
      const auto clamp_x = [&](float value) {
        return std::clamp(value * scale_x, 0.0F,
                          static_cast<float>(vehicle.width - 1));
      };
      const auto clamp_y = [&](float value) {
        return std::clamp(value * scale_y, 0.0F,
                          static_cast<float>(vehicle.height - 1));
      };
      plate_views.push_back({vehicle,
                             clamp_x(boxes[i].x0), clamp_y(boxes[i].y0),
                             clamp_x(boxes[i].x1), clamp_y(boxes[i].y1),
                             clamp_x(boxes[i].x2), clamp_y(boxes[i].y2),
                             clamp_x(boxes[i].x3), clamp_y(boxes[i].y3)});
      owners.push_back(offset + static_cast<std::size_t>(i));
    }
  }
  static constexpr float rec_mean[3] = {0.5F, 0.5F, 0.5F};
  static constexpr float rec_inverse_std[3] = {2.0F, 2.0F, 2.0F};
  constexpr int max_tokens = 32;
  for (std::size_t offset = 0; offset < plate_views.size(); offset += kRoiChunk) {
    const int count = static_cast<int>(std::min(kRoiChunk, plate_views.size() - offset));
    roi_input_.resize(static_cast<std::size_t>(count) * 3 * kPlateRecHeight *
                      kPlateRecWidth * sizeof(float));
    launch_warp_quad_normalize(
        plate_views.data() + offset, count,
        static_cast<float*>(roi_input_.data()), kPlateRecWidth,
        kPlateRecHeight, rec_mean, rec_inverse_std, true, true, 127.5F,
        primary_stream_);
    auto recognition_outputs = plate_recognizer_->infer(
        {{"x", roi_input_.data(), {count, 3, kPlateRecHeight, kPlateRecWidth},
          nvinfer1::DataType::kFLOAT, roi_input_.capacity()}}, primary_stream_);
    const auto& probabilities = only_output(recognition_outputs);
    if (probabilities.shape.size() != 3) {
      throw std::runtime_error("PP-OCRv3 recognizer output contract mismatch");
    }
    tokens_device_.resize(static_cast<std::size_t>(count) * max_tokens * sizeof(int));
    token_counts_device_.resize(static_cast<std::size_t>(count) * sizeof(int));
    launch_ctc_argmax(static_cast<const float*>(probabilities.data), count,
                      static_cast<int>(probabilities.shape[1]),
                      static_cast<int>(probabilities.shape[2]), max_tokens,
                      static_cast<int*>(tokens_device_.data()),
                      static_cast<int*>(token_counts_device_.data()), primary_stream_);
    std::vector<int> tokens(static_cast<std::size_t>(count) * max_tokens);
    std::vector<int> counts(count);
    check_cuda(cudaMemcpyAsync(tokens.data(), tokens_device_.data(),
                               tokens.size() * sizeof(int), cudaMemcpyDeviceToHost,
                               primary_stream_), "copy OCR tokens");
    check_cuda(cudaMemcpyAsync(counts.data(), token_counts_device_.data(),
                               counts.size() * sizeof(int), cudaMemcpyDeviceToHost,
                               primary_stream_), "copy OCR token counts");
    check_cuda(cudaStreamSynchronize(primary_stream_), "synchronize plate recognition");
    const std::string provinces =
        "浙粤京津冀晋蒙辽黑沪吉苏皖赣鲁豫鄂湘桂琼渝川贵云藏陕甘青宁闽";
    for (int row = 0; row < count; ++row) {
      std::string text;
      int accepted = 0;
      for (int i = 0; i < std::min(counts[row], max_tokens); ++i) {
        const int token = tokens[static_cast<std::size_t>(row) * max_tokens + i];
        if (token <= 0 || static_cast<std::size_t>(token) >= characters_.size()) continue;
        std::string value = characters_[static_cast<std::size_t>(token)];
        const bool ascii = value.size() == 1 &&
                           std::isalnum(static_cast<unsigned char>(value[0]));
        if (ascii) value[0] = static_cast<char>(std::toupper(value[0]));
        if (ascii || provinces.find(value) != std::string::npos) {
          text += value;
          ++accepted;
        }
      }
      if (accepted > 2 && accepted < 10) {
        result[owners[offset + static_cast<std::size_t>(row)]] = std::move(text);
      }
    }
  }
  return result;
}

std::vector<PipelineResult> Pipeline::Impl::run(const std::vector<Task>& tasks) {
  decoder_.decode(tasks, decoded_pool_, primary_stream_);
  auto& decoded = decoded_pool_;
  check_cuda(cudaStreamSynchronize(primary_stream_), "synchronize image decode");
  std::vector<int> valid_to_task;
  std::vector<ImageView> image_views;
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    if (decoded[i].valid) {
      valid_to_task.push_back(static_cast<int>(i));
      image_views.push_back({static_cast<const std::uint8_t*>(decoded[i].pixels.data()),
                             decoded[i].width, decoded[i].height,
                             decoded[i].pitch});
    }
  }
  std::vector<PipelineResult> output(tasks.size());
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    if (!decoded[i].valid) {
      output[i] = {.ok = false, .error = decoded[i].error};
    }
  }
  if (image_views.empty()) return output;

  const int batch = static_cast<int>(image_views.size());
  detector_input_.resize(static_cast<std::size_t>(batch) * 3 * kDetectorSide *
                         kDetectorSide * sizeof(float));
  // Preserve the deployed v1 PaddleDetector input contract exactly: resized
  // RGB CHW float32 in the original [0, 255] range.  Normalization belongs to
  // the attribute/OCR models, not these already exported detector graphs.
  static constexpr float detector_mean[3] = {0.0F, 0.0F, 0.0F};
  static constexpr float detector_inverse_std[3] = {1.0F, 1.0F, 1.0F};
  launch_resize_normalize(image_views.data(), batch,
                          static_cast<float*>(detector_input_.data()),
                          kDetectorSide, kDetectorSide, detector_mean,
                          detector_inverse_std, false, false, ResizeMode::stretch,
                          Interpolation::cubic, 0.0F, primary_stream_);
  std::vector<float> scale(static_cast<std::size_t>(batch) * 2);
  for (int i = 0; i < batch; ++i) {
    scale[static_cast<std::size_t>(i) * 2] =
        static_cast<float>(kDetectorSide) / image_views[i].height;
    scale[static_cast<std::size_t>(i) * 2 + 1] =
        static_cast<float>(kDetectorSide) / image_views[i].width;
  }
  scale_input_.resize(scale.size() * sizeof(float));
  check_cuda(cudaMemcpyAsync(scale_input_.data(), scale.data(),
                             scale.size() * sizeof(float), cudaMemcpyHostToDevice,
                             primary_stream_), "copy detector scale factors");
  cudaEvent_t ready_event = nullptr;
  check_cuda(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming),
             "cudaEventCreateWithFlags");
  check_cuda(cudaEventRecord(ready_event, primary_stream_), "cudaEventRecord");
  check_cuda(cudaStreamWaitEvent(secondary_stream_, ready_event),
             "cudaStreamWaitEvent");
  const DeviceTensor image_input{"image", detector_input_.data(),
                                 {batch, 3, kDetectorSide, kDetectorSide},
                                 nvinfer1::DataType::kFLOAT,
                                 detector_input_.capacity()};
  const DeviceTensor scale_tensor{"scale_factor", scale_input_.data(), {batch, 2},
                                  nvinfer1::DataType::kFLOAT,
                                  scale_input_.capacity()};
  enqueue_detect(*person_detector_, image_input, scale_tensor, batch,
                 person_nms_boxes_, person_nms_counts_, primary_stream_);
  enqueue_detect(*vehicle_detector_, image_input, scale_tensor, batch,
                 vehicle_nms_boxes_, vehicle_nms_counts_, secondary_stream_);
  auto person_boxes = collect_detections(
      batch, person_nms_boxes_, person_nms_counts_, primary_stream_);
  auto vehicle_boxes = collect_detections(
      batch, vehicle_nms_boxes_, vehicle_nms_counts_, secondary_stream_);
  cudaEventDestroy(ready_event);

  std::vector<RoiJob> person_jobs;
  std::vector<RoiJob> vehicle_jobs;
  for (const auto& box : person_boxes) {
    const auto& image = decoded[static_cast<std::size_t>(valid_to_task[box.image_index])];
    const auto body = expanded_crop_view(image, box, 1.0F);
    if (!body) continue;
    person_jobs.push_back(
        {box.image_index, box, *body, upper_crop_view(image, box, 0.40F)});
  }
  for (const auto& box : vehicle_boxes) {
    const auto& image = decoded[static_cast<std::size_t>(valid_to_task[box.image_index])];
    const auto body = expanded_crop_view(image, box, 1.3F);
    if (!body) continue;
    vehicle_jobs.push_back({box.image_index, box, *body, std::nullopt});
  }
  std::vector<ImageView> person_bodies;
  std::vector<ImageView> person_heads;
  std::vector<std::size_t> person_head_owners;
  std::vector<ImageView> vehicle_bodies;
  for (std::size_t i = 0; i < person_jobs.size(); ++i) {
    const auto& job = person_jobs[i];
    person_bodies.push_back(job.body);
    if (job.head) {
      person_heads.push_back(*job.head);
      person_head_owners.push_back(i);
    }
  }
  for (const auto& job : vehicle_jobs) vehicle_bodies.push_back(job.body);
  auto person_scores = attributes(*person_attribute_, person_bodies, 192, 256, 26);
  std::vector<int> mask_labels(person_jobs.size(), 0);
  const auto detected_mask_labels = masks(person_heads);
  for (std::size_t i = 0; i < person_head_owners.size(); ++i) {
    mask_labels[person_head_owners[i]] = detected_mask_labels[i];
  }
  auto vehicle_scores = attributes(*vehicle_attribute_, vehicle_bodies, 256, 192, 19);
  auto plate_texts = plates(vehicle_bodies);

  std::vector<std::vector<std::string>> people(batch);
  std::vector<std::vector<std::string>> vehicles(batch);
  std::vector<std::vector<std::size_t>> kept_vehicle_indices(batch);
  for (std::size_t i = 0; i < person_jobs.size(); ++i) {
    people[person_jobs[i].image].push_back(
        decode_person(person_scores.data() + i * 26, mask_labels[i]));
  }
  for (std::size_t i = 0; i < vehicle_jobs.size(); ++i) {
    const auto image = static_cast<std::size_t>(vehicle_jobs[i].image);
    const bool duplicate = plate_texts[i] != "未识别" && std::any_of(
        kept_vehicle_indices[image].begin(), kept_vehicle_indices[image].end(),
        [&](std::size_t previous) {
          return plate_texts[previous] == plate_texts[i] &&
                 contained_overlap(vehicle_jobs[previous].detection,
                                   vehicle_jobs[i].detection) >= 0.5F;
        });
    if (duplicate) continue;
    kept_vehicle_indices[image].push_back(i);
    vehicles[image].push_back(
        decode_vehicle(vehicle_scores.data() + i * 19, plate_texts[i]));
  }
  for (int image = 0; image < batch; ++image) {
    std::ostringstream json;
    json << "{\"行人\":[";
    for (std::size_t i = 0; i < people[image].size(); ++i) {
      if (i) json << ',';
      json << people[image][i];
    }
    json << "],\"车辆\":[";
    for (std::size_t i = 0; i < vehicles[image].size(); ++i) {
      if (i) json << ',';
      json << vehicles[image][i];
    }
    json << "]}";
    output[static_cast<std::size_t>(valid_to_task[image])] =
        {.ok = true, .json = json.str()};
  }
  return output;
}

Pipeline::Pipeline(const Config& config) : impl_(std::make_unique<Impl>(config)) {}
Pipeline::~Pipeline() = default;

std::vector<PipelineResult> Pipeline::run(const std::vector<Task>& tasks) {
  return impl_->run(tasks);
}

std::unordered_map<std::string, std::string> Pipeline::health() const {
  return impl_->health();
}

}  // namespace pvr
