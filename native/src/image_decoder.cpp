#include "pvr/image_decoder.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include <stb/stb_image.h>
#include <webp/decode.h>

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvr {
namespace {

void check_nvjpeg(nvjpegStatus_t status, const char* operation) {
  if (status != NVJPEG_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with status " +
                             std::to_string(static_cast<int>(status)));
  }
}

void validate_dimensions(int width, int height, std::size_t max_pixels) {
  if (width <= 0 || height <= 0 ||
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) >
          max_pixels) {
    throw std::runtime_error("decoded image dimensions exceed limit");
  }
}

}  // namespace

ImageDecoder::ImageDecoder(std::size_t max_pixels) : max_pixels_(max_pixels) {
  check_nvjpeg(nvjpegCreateSimple(&handle_), "nvjpegCreateSimple");
  check_nvjpeg(nvjpegJpegStateCreate(handle_, &state_),
               "nvjpegJpegStateCreate");
}

ImageDecoder::~ImageDecoder() {
  if (state_) {
    nvjpegJpegStateDestroy(state_);
  }
  if (handle_) {
    nvjpegDestroy(handle_);
  }
}

void ImageDecoder::recreate_state() {
  if (state_) {
    nvjpegJpegStateDestroy(state_);
    state_ = nullptr;
  }
  check_nvjpeg(nvjpegJpegStateCreate(handle_, &state_),
               "nvjpegJpegStateCreate");
}

void ImageDecoder::decode_jpeg_batch(const std::vector<Task>& tasks,
                                     std::vector<DeviceImage>& output,
                                     cudaStream_t stream) {
  std::vector<std::size_t> indexes;
  std::vector<const unsigned char*> bitstreams;
  std::vector<std::size_t> lengths;
  std::vector<nvjpegImage_t> destinations;
  indexes.reserve(tasks.size());
  bitstreams.reserve(tasks.size());
  lengths.reserve(tasks.size());
  destinations.reserve(tasks.size());

  for (std::size_t index = 0; index < tasks.size(); ++index) {
    if (tasks[index].media_type != MediaType::jpeg) {
      continue;
    }
    try {
      int components = 0;
      nvjpegChromaSubsampling_t subsampling{};
      std::array<int, NVJPEG_MAX_COMPONENT> widths{};
      std::array<int, NVJPEG_MAX_COMPONENT> heights{};
      check_nvjpeg(nvjpegGetImageInfo(
                       handle_, tasks[index].payload->data(),
                       tasks[index].payload->size(), &components, &subsampling,
                       widths.data(), heights.data()),
                   "nvjpegGetImageInfo");
      validate_dimensions(widths[0], heights[0], max_pixels_);
      auto& image = output[index];
      image.width = widths[0];
      image.height = heights[0];
      image.pitch = static_cast<std::size_t>(image.width) * 3;
      image.pixels.resize(image.pitch * static_cast<std::size_t>(image.height));
      nvjpegImage_t destination{};
      destination.channel[0] =
          static_cast<unsigned char*>(image.pixels.data());
      destination.pitch[0] = image.pitch;
      indexes.push_back(index);
      bitstreams.push_back(tasks[index].payload->data());
      lengths.push_back(tasks[index].payload->size());
      destinations.push_back(destination);
    } catch (const std::exception& error) {
      output[index].error = error.what();
    }
  }
  if (indexes.empty()) {
    return;
  }

  const int batch = static_cast<int>(indexes.size());
  const int cpu_threads = std::min(batch, 8);
  auto status = nvjpegDecodeBatchedInitialize(handle_, state_, batch,
                                               cpu_threads, NVJPEG_OUTPUT_RGBI);
  if (status == NVJPEG_STATUS_SUCCESS) {
    status = nvjpegDecodeBatched(handle_, state_, bitstreams.data(),
                                 lengths.data(), destinations.data(), stream);
  }
  if (status == NVJPEG_STATUS_SUCCESS) {
    for (const auto index : indexes) {
      output[index].valid = true;
    }
    return;
  }

  // A malformed stream must not poison its neighbors. Reset the persistent
  // decoder state and isolate failures only when the batch call rejects.
  check_cuda(cudaStreamSynchronize(stream),
             "synchronize failed nvJPEG batch");
  recreate_state();
  for (const auto index : indexes) {
    try {
      decode_jpeg_one(tasks[index], output[index], stream);
    } catch (const std::exception& error) {
      output[index].valid = false;
      output[index].error = error.what();
    }
  }
}

void ImageDecoder::decode_jpeg_one(const Task& task, DeviceImage& output,
                                   cudaStream_t stream) {
  int components = 0;
  nvjpegChromaSubsampling_t subsampling{};
  std::array<int, NVJPEG_MAX_COMPONENT> widths{};
  std::array<int, NVJPEG_MAX_COMPONENT> heights{};
  check_nvjpeg(
      nvjpegGetImageInfo(handle_, task.payload->data(), task.payload->size(),
                         &components, &subsampling, widths.data(), heights.data()),
      "nvjpegGetImageInfo");
  validate_dimensions(widths[0], heights[0], max_pixels_);
  output.width = widths[0];
  output.height = heights[0];
  output.pitch = static_cast<std::size_t>(output.width) * 3;
  output.pixels.resize(output.pitch * static_cast<std::size_t>(output.height));
  nvjpegImage_t destination{};
  destination.channel[0] = static_cast<unsigned char*>(output.pixels.data());
  destination.pitch[0] = output.pitch;
  check_nvjpeg(nvjpegDecode(handle_, state_, task.payload->data(),
                            task.payload->size(), NVJPEG_OUTPUT_RGBI,
                            &destination, stream),
               "nvjpegDecode");
  output.valid = true;
}

void ImageDecoder::decode_compat_one(const Task& task, DeviceImage& output,
                                     cudaStream_t stream) {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::unique_ptr<std::uint8_t, decltype(&stbi_image_free)> decoded(
      nullptr, &stbi_image_free);
  std::uint8_t* webp = nullptr;
  if (task.media_type == MediaType::webp) {
    webp = WebPDecodeRGB(task.payload->data(), task.payload->size(), &width,
                         &height);
    if (!webp) {
      throw std::runtime_error("WebP decode failed");
    }
  } else {
    decoded.reset(stbi_load_from_memory(
        task.payload->data(), static_cast<int>(task.payload->size()), &width,
        &height, &channels, 3));
    if (!decoded) {
      throw std::runtime_error("PNG/BMP decode failed");
    }
  }
  std::unique_ptr<std::uint8_t, decltype(&WebPFree)> webp_owner(webp, &WebPFree);
  validate_dimensions(width, height, max_pixels_);
  output.width = width;
  output.height = height;
  output.pitch = static_cast<std::size_t>(width) * 3;
  output.pixels.resize(output.pitch * static_cast<std::size_t>(height));
  const auto* source = webp ? webp : decoded.get();
  check_cuda(cudaMemcpyAsync(output.pixels.data(), source,
                             output.pitch * static_cast<std::size_t>(height),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync compatibility image");
  output.valid = true;
}

void ImageDecoder::decode(const std::vector<Task>& tasks,
                          std::vector<DeviceImage>& output,
                          cudaStream_t stream) {
  // Grow to the largest observed batch and retain every device allocation.
  // Resetting metadata is cheap; DeviceBuffer::resize only reallocates when a
  // later image exceeds that slot's historical high-water mark.
  if (output.size() < tasks.size()) output.resize(tasks.size());
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    output[i].width = 0;
    output[i].height = 0;
    output[i].pitch = 0;
    output[i].valid = false;
    output[i].error.clear();
  }
  decode_jpeg_batch(tasks, output, stream);
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    if (tasks[i].media_type == MediaType::jpeg) {
      continue;
    }
    try {
      decode_compat_one(tasks[i], output[i], stream);
    } catch (const std::exception& error) {
      output[i].valid = false;
      output[i].error = error.what();
    }
  }
}

}  // namespace pvr
