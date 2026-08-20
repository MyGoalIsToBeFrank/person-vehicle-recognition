#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace pvr {

inline void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes) { resize(bytes); }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        capacity_(std::exchange(other.capacity_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      cudaFree(data_);
      data_ = std::exchange(other.data_, nullptr);
      capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
  }

  void resize(std::size_t bytes) {
    if (bytes <= capacity_) {
      return;
    }
    void* replacement = nullptr;
    check_cuda(cudaMalloc(&replacement, bytes), "cudaMalloc");
    cudaFree(data_);
    data_ = replacement;
    capacity_ = bytes;
  }
  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  void* data_ = nullptr;
  std::size_t capacity_ = 0;
};

class CudaStream {
 public:
  CudaStream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }
  ~CudaStream() { cudaStreamDestroy(stream_); }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  operator cudaStream_t() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

}  // namespace pvr
