#pragma once

#include "pvr/cuda_utils.hpp"

#include <NvInfer.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pvr {

struct ShapeRange {
  std::vector<std::int64_t> min;
  std::vector<std::int64_t> opt;
  std::vector<std::int64_t> max;
};

struct DeviceTensor {
  std::string name;
  void* data = nullptr;
  std::vector<std::int64_t> shape;
  nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
  std::size_t bytes = 0;
};

class TensorRtModel {
 public:
  TensorRtModel(const std::filesystem::path& onnx,
                const std::filesystem::path& cache_dir,
                std::unordered_map<std::string, ShapeRange> profiles,
                std::string precision);
  ~TensorRtModel();
  TensorRtModel(const TensorRtModel&) = delete;
  TensorRtModel& operator=(const TensorRtModel&) = delete;

  std::vector<DeviceTensor> infer(const std::vector<DeviceTensor>& inputs,
                                  cudaStream_t stream);
  std::size_t context_memory_bytes() const noexcept;
  void bind_context_memory(void* memory, std::size_t bytes);
  const std::string& precision() const noexcept { return precision_; }
  const std::string& cache_key() const noexcept { return cache_key_; }

 private:
  class Logger;
  struct CapturedGraph;
  void load_or_build(const std::filesystem::path& onnx,
                     const std::filesystem::path& cache_dir,
                     const std::unordered_map<std::string, ShapeRange>& profiles);

  std::unique_ptr<Logger> logger_;
  struct RuntimeDeleter {
    void operator()(nvinfer1::IRuntime* value) const { delete value; }
  };
  struct EngineDeleter {
    void operator()(nvinfer1::ICudaEngine* value) const { delete value; }
  };
  struct ContextDeleter {
    void operator()(nvinfer1::IExecutionContext* value) const { delete value; }
  };
  std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> context_;
  std::unordered_map<std::string, DeviceBuffer> outputs_;
  std::unordered_map<std::string, std::unique_ptr<CapturedGraph>> graphs_;
  std::unordered_set<std::string> graph_fallbacks_;
  bool context_memory_bound_ = false;
  std::string precision_;
  std::string cache_key_;
};

std::size_t tensor_bytes(nvinfer1::Dims dims, nvinfer1::DataType type);

}  // namespace pvr
