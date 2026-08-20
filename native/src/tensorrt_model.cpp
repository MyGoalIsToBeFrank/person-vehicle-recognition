#include "pvr/tensorrt_model.hpp"

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace pvr {
namespace {

static_assert(NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR == 5,
              "PVR v2 requires TensorRT 10.5 exactly");

template <typename T>
struct TrtDeleter {
  void operator()(T* value) const { delete value; }
};

nvinfer1::Dims to_dims(const std::vector<std::int64_t>& shape) {
  if (shape.size() > nvinfer1::Dims::MAX_DIMS) {
    throw std::invalid_argument("TensorRT shape has too many dimensions");
  }
  nvinfer1::Dims dims;
  dims.nbDims = static_cast<int>(shape.size());
  for (int i = 0; i < dims.nbDims; ++i) {
    dims.d[i] = shape[static_cast<std::size_t>(i)];
  }
  return dims;
}

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open ONNX model: " + path.string());
  }
  SHA256_CTX context;
  SHA256_Init(&context);
  std::array<char, 1 << 20> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    SHA256_Update(&context, buffer.data(),
                  static_cast<std::size_t>(input.gcount()));
  }
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  SHA256_Final(digest.data(), &context);
  std::ostringstream output;
  for (const auto byte : digest) {
    output << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<int>(byte);
  }
  return output.str();
}

std::string sha256_text(const std::string& value) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  SHA256(reinterpret_cast<const unsigned char*>(value.data()), value.size(),
         digest.data());
  std::ostringstream output;
  for (const auto byte : digest) {
    output << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<int>(byte);
  }
  return output.str();
}

std::string profile_identity(
    const std::unordered_map<std::string, ShapeRange>& profiles) {
  std::vector<std::string> names;
  names.reserve(profiles.size());
  for (const auto& [name, unused] : profiles) {
    static_cast<void>(unused);
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  std::ostringstream serialized;
  for (const auto& name : names) {
    serialized << name;
    const auto& range = profiles.at(name);
    for (const auto* shape : {&range.min, &range.opt, &range.max}) {
      serialized << ':';
      for (const auto dimension : *shape) {
        serialized << dimension << ',';
      }
    }
    serialized << ';';
  }
  return sha256_text(serialized.str());
}

std::string gpu_identity() {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device),
             "cudaGetDeviceProperties");
  std::string name = properties.name;
  std::replace_if(name.begin(), name.end(),
                  [](unsigned char value) { return !std::isalnum(value); }, '_');
  return name + "_sm" + std::to_string(properties.major) +
         std::to_string(properties.minor);
}

bool graph_batch(int batch) {
  switch (batch) {
    case 1:
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
      return true;
    default:
      return false;
  }
}

std::string graph_identity(const std::vector<DeviceTensor>& inputs,
                           const std::vector<DeviceTensor>& outputs) {
  if (inputs.empty() || inputs.front().shape.empty() ||
      !graph_batch(static_cast<int>(inputs.front().shape.front()))) {
    return {};
  }
  const auto batch = inputs.front().shape.front();
  std::ostringstream serialized;
  serialized << "batch=" << batch;
  const auto append = [&](const DeviceTensor& tensor) {
    if (tensor.shape.empty() || tensor.shape.front() != batch) {
      return false;
    }
    serialized << ';' << tensor.name << '@' << tensor.data << ':';
    for (const auto dimension : tensor.shape) {
      serialized << dimension << ',';
    }
    return true;
  };
  for (const auto& input : inputs) {
    if (!append(input)) return {};
  }
  for (const auto& output : outputs) {
    if (!append(output)) return {};
  }
  return serialized.str();
}

std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void atomic_write(const std::filesystem::path& target, const void* data,
                  std::size_t size) {
  std::filesystem::create_directories(target.parent_path());
  const auto temporary = target.string() + ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.write(static_cast<const char*>(data),
                 static_cast<std::streamsize>(size));
    if (!output) {
      throw std::runtime_error("cannot write TensorRT engine cache");
    }
  }
  std::filesystem::rename(temporary, target);
}

}  // namespace

class TensorRtModel::Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << message << '\n';
    }
  }
};

struct TensorRtModel::CapturedGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;

  ~CapturedGraph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
};

std::size_t tensor_bytes(nvinfer1::Dims dims, nvinfer1::DataType type) {
  std::size_t elements = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      throw std::runtime_error("unresolved TensorRT output dimension");
    }
    elements *= static_cast<std::size_t>(dims.d[i]);
  }
  std::size_t element_bytes = 0;
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      element_bytes = 4;
      break;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
      element_bytes = 2;
      break;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
      element_bytes = 1;
      break;
    case nvinfer1::DataType::kINT64:
      element_bytes = 8;
      break;
    default:
      throw std::runtime_error("unsupported packed TensorRT tensor type");
  }
  return elements * element_bytes;
}

TensorRtModel::TensorRtModel(
    const std::filesystem::path& onnx,
    const std::filesystem::path& cache_dir,
    std::unordered_map<std::string, ShapeRange> profiles, std::string precision)
    : logger_(std::make_unique<Logger>()), precision_(std::move(precision)) {
  if (precision_ != "fp16" && precision_ != "int8") {
    throw std::invalid_argument("precision must be fp16 or int8");
  }
  load_or_build(onnx, cache_dir, profiles);
}

TensorRtModel::~TensorRtModel() = default;

void TensorRtModel::load_or_build(
    const std::filesystem::path& onnx,
    const std::filesystem::path& cache_dir,
    const std::unordered_map<std::string, ShapeRange>& profiles) {
  initLibNvInferPlugins(logger_.get(), "");
  std::size_t free_memory = 0;
  std::size_t total_memory = 0;
  check_cuda(cudaMemGetInfo(&free_memory, &total_memory), "cudaMemGetInfo");
  static_cast<void>(free_memory);
  constexpr int kBuilderOptimizationLevel = 5;
  const std::size_t workspace_limit =
      total_memory >= (20ULL << 30) ? (12ULL << 30) : (6ULL << 30);
  const auto digest = sha256_file(onnx);
  cache_key_ = gpu_identity() + "_trt" + std::to_string(NV_TENSORRT_MAJOR) +
               "." + std::to_string(NV_TENSORRT_MINOR) + "." +
               std::to_string(NV_TENSORRT_PATCH) + "_" + precision_ + "_" +
               digest + "_bo" + std::to_string(kBuilderOptimizationLevel) +
               "_ws" + std::to_string(workspace_limit >> 30) + "g_profile" +
               profile_identity(profiles);
  const auto cache_file = cache_dir / (onnx.stem().string() + "_" + cache_key_ + ".engine");
  auto plan = read_file(cache_file);
  const bool loaded_from_cache = !plan.empty();

  if (plan.empty()) {
    std::cerr << "[PVR] building TensorRT engine " << onnx.stem().string()
              << " -> " << cache_file.filename().string() << '\n';
    std::unique_ptr<nvinfer1::IBuilder, TrtDeleter<nvinfer1::IBuilder>> builder(
        nvinfer1::createInferBuilder(*logger_));
    if (!builder) {
      throw std::runtime_error("createInferBuilder failed");
    }
    const std::uint32_t flags = 0;
    std::unique_ptr<nvinfer1::INetworkDefinition,
                    TrtDeleter<nvinfer1::INetworkDefinition>>
        network(builder->createNetworkV2(flags));
    std::unique_ptr<nvonnxparser::IParser, TrtDeleter<nvonnxparser::IParser>> parser(
        nvonnxparser::createParser(*network, *logger_));
    if (!parser || !parser->parseFromFile(
                       onnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      throw std::runtime_error("TensorRT failed to parse " + onnx.string());
    }
    std::unique_ptr<nvinfer1::IBuilderConfig,
                    TrtDeleter<nvinfer1::IBuilderConfig>>
        build_config(builder->createBuilderConfig());
    build_config->setBuilderOptimizationLevel(kBuilderOptimizationLevel);
    build_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                     workspace_limit);
    if (precision_ == "fp16") {
      build_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
      build_config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }
    auto* profile = builder->createOptimizationProfile();
    for (const auto& [name, range] : profiles) {
      if (!profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
                                  to_dims(range.min)) ||
          !profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
                                  to_dims(range.opt)) ||
          !profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
                                  to_dims(range.max))) {
        throw std::runtime_error("invalid TensorRT profile for input " + name);
      }
    }
    if (build_config->addOptimizationProfile(profile) < 0) {
      throw std::runtime_error("TensorRT rejected optimization profile");
    }
    std::unique_ptr<nvinfer1::IHostMemory, TrtDeleter<nvinfer1::IHostMemory>> serialized(
        builder->buildSerializedNetwork(*network, *build_config));
    if (!serialized) {
      throw std::runtime_error("TensorRT engine build failed for " + onnx.string());
    }
    atomic_write(cache_file, serialized->data(), serialized->size());
    std::cerr << "[PVR] built TensorRT engine " << onnx.stem().string()
              << " (" << serialized->size() << " bytes)\n";
    plan.assign(static_cast<const char*>(serialized->data()),
                static_cast<const char*>(serialized->data()) + serialized->size());
  } else {
    std::cerr << "[PVR] TensorRT cache hit " << onnx.stem().string()
              << " (" << plan.size() << " bytes)\n";
  }

  runtime_.reset(nvinfer1::createInferRuntime(*logger_));
  engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine_) {
    if (loaded_from_cache) {
      std::filesystem::remove(cache_file);
      load_or_build(onnx, cache_dir, profiles);
      return;
    }
    throw std::runtime_error("TensorRT engine deserialization failed: " +
                             cache_file.string());
  }
  context_.reset(engine_->createExecutionContext(
      nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if (!context_) {
    throw std::runtime_error("TensorRT execution context creation failed");
  }
}

std::size_t TensorRtModel::context_memory_bytes() const noexcept {
  const auto bytes = engine_->getDeviceMemorySizeV2();
  return bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
}

void TensorRtModel::bind_context_memory(void* memory, std::size_t bytes) {
  const auto required = context_memory_bytes();
  if (bytes < required || (required > 0 && memory == nullptr)) {
    throw std::invalid_argument("TensorRT context memory pool is too small");
  }
  context_->setDeviceMemoryV2(memory, static_cast<std::int64_t>(bytes));
  context_memory_bound_ = true;
  std::cerr << "[PVR] TensorRT context memory bound (" << required
            << " bytes required, " << bytes << " bytes supplied)\n";
}

std::vector<DeviceTensor> TensorRtModel::infer(
    const std::vector<DeviceTensor>& inputs, cudaStream_t stream) {
  if (!context_memory_bound_) {
    throw std::runtime_error("TensorRT context memory is not bound");
  }
  for (const auto& input : inputs) {
    if (!context_->setInputShape(input.name.c_str(), to_dims(input.shape)) ||
        !context_->setTensorAddress(input.name.c_str(), input.data)) {
      throw std::runtime_error("failed to bind TensorRT input " + input.name);
    }
  }
  if (!context_->allInputDimensionsSpecified()) {
    throw std::runtime_error("not all TensorRT input dimensions were specified");
  }

  std::vector<DeviceTensor> output;
  const int count = engine_->getNbIOTensors();
  for (int i = 0; i < count; ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT) {
      continue;
    }
    const auto dims = context_->getTensorShape(name);
    const auto type = engine_->getTensorDataType(name);
    const auto bytes = tensor_bytes(dims, type);
    auto& buffer = outputs_[name];
    buffer.resize(bytes);
    if (!context_->setTensorAddress(name, buffer.data())) {
      throw std::runtime_error("failed to bind TensorRT output " +
                               std::string(name));
    }
    std::vector<std::int64_t> shape(dims.nbDims);
    for (int dim = 0; dim < dims.nbDims; ++dim) {
      shape[static_cast<std::size_t>(dim)] = dims.d[dim];
    }
    output.push_back(DeviceTensor{.name = name,
                                  .data = buffer.data(),
                                  .shape = std::move(shape),
                                  .type = type,
                                  .bytes = bytes});
  }
  const auto graph_key = graph_identity(inputs, output);
  if (!graph_key.empty()) {
    const auto captured = graphs_.find(graph_key);
    if (captured != graphs_.end()) {
      check_cuda(cudaGraphLaunch(captured->second->executable, stream),
                 "cudaGraphLaunch TensorRT");
      return output;
    }
  }

  if (!context_->enqueueV3(stream)) {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }

  // Dynamic-shape TensorRT contexts must execute once after a shape change
  // before capture. Capture only stable, high-value batch buckets, and bind
  // the exact input/output addresses into the key so a growing memory pool can
  // never launch a graph that references an old allocation.
  if (!graph_key.empty() && graphs_.size() < 32 &&
      !graph_fallbacks_.contains(graph_key)) {
    check_cuda(cudaStreamSynchronize(stream),
               "synchronize TensorRT CUDA Graph warmup");
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) ==
        cudaSuccess) {
      const bool enqueued = context_->enqueueV3(stream);
      cudaGraph_t graph = nullptr;
      const auto ended = cudaStreamEndCapture(stream, &graph);
      if (enqueued && ended == cudaSuccess && graph) {
        cudaGraphExec_t executable = nullptr;
        const auto instantiated =
            cudaGraphInstantiate(&executable, graph, 0);
        if (instantiated == cudaSuccess) {
          auto captured = std::make_unique<CapturedGraph>();
          captured->graph = graph;
          captured->executable = executable;
          graphs_.emplace(graph_key, std::move(captured));
        } else {
          cudaGraphDestroy(graph);
          graph_fallbacks_.insert(graph_key);
        }
      } else {
        if (graph) cudaGraphDestroy(graph);
        graph_fallbacks_.insert(graph_key);
        cudaGetLastError();
      }
    } else {
      graph_fallbacks_.insert(graph_key);
      cudaGetLastError();
    }
  }
  return output;
}

}  // namespace pvr
