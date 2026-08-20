#pragma once

#include "pvr/cuda_utils.hpp"
#include "pvr/types.hpp"

#include <nvjpeg.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pvr {

struct DeviceImage {
  DeviceBuffer pixels;
  int width = 0;
  int height = 0;
  std::size_t pitch = 0;
  bool valid = false;
  std::string error;
};

class ImageDecoder {
 public:
  explicit ImageDecoder(std::size_t max_pixels);
  ~ImageDecoder();
  ImageDecoder(const ImageDecoder&) = delete;
  ImageDecoder& operator=(const ImageDecoder&) = delete;

  void decode(const std::vector<Task>& tasks, std::vector<DeviceImage>& output,
              cudaStream_t stream);

 private:
  void decode_jpeg_batch(const std::vector<Task>& tasks,
                         std::vector<DeviceImage>& output,
                         cudaStream_t stream);
  void decode_jpeg_one(const Task& task, DeviceImage& output,
                       cudaStream_t stream);
  void decode_compat_one(const Task& task, DeviceImage& output,
                         cudaStream_t stream);
  void recreate_state();

  std::size_t max_pixels_;
  nvjpegHandle_t handle_ = nullptr;
  nvjpegJpegState_t state_ = nullptr;
};

}  // namespace pvr
