#include "pvr/engine.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

pvr::Config parse_config(const py::dict& input) {
  pvr::Config config;
  config.model_dir = py::cast<std::string>(input["model_dir"]);
  config.engine_cache_dir =
      py::cast<std::string>(input["engine_cache_dir"]);
  config.max_image_bytes = py::cast<std::size_t>(input["max_image_bytes"]);
  config.max_image_pixels = py::cast<std::size_t>(input["max_image_pixels"]);
  config.max_batch_images = py::cast<std::size_t>(input["max_batch_images"]);
  config.max_queue_images = py::cast<std::size_t>(input["max_queue_images"]);
  config.max_queue_bytes = py::cast<std::size_t>(input["max_queue_bytes"]);
  config.batch_wait =
      std::chrono::microseconds(py::cast<int>(input["batch_wait_us"]));
  config.result_ttl =
      std::chrono::seconds(py::cast<int>(input["result_ttl_seconds"]));
  config.max_result_bytes = py::cast<std::size_t>(input["max_result_bytes"]);
  config.max_result_records =
      py::cast<std::size_t>(input["max_result_records"]);
  return config;
}

pvr::MediaType parse_media_type(int value) {
  if (value < static_cast<int>(pvr::MediaType::jpeg) ||
      value > static_cast<int>(pvr::MediaType::webp)) {
    throw std::invalid_argument("invalid media type");
  }
  return static_cast<pvr::MediaType>(value);
}

std::vector<std::uint8_t> copy_bytes(const py::bytes& payload) {
  std::string value = payload;
  return {value.begin(), value.end()};
}

py::dict record_to_dict(const pvr::Record& record) {
  py::dict output;
  output["status"] = pvr::status_name(record.status);
  if (record.status == pvr::Status::done) {
    output["result"] = py::module_::import("json").attr("loads")(
        record.result_json);
  } else if (record.status == pvr::Status::error) {
    output["error"] = record.error;
  }
  if (record.status == pvr::Status::done || record.status == pvr::Status::error) {
    output["timing_ms"] = py::dict(
        "queue"_a = record.queue_ms, "inference"_a = record.inference_ms,
        "total"_a = record.total_ms);
  }
  return output;
}

}  // namespace

PYBIND11_MODULE(pvr_native, module) {
  module.doc() = "Bounded single-instance CUDA/TensorRT PVR worker";
  py::register_exception<pvr::QueueFullError>(module, "QueueFullError");
  py::register_exception<pvr::NotReadyError>(module, "NotReadyError");
  py::register_exception<pvr::PayloadTooLargeError>(module,
                                                    "PayloadTooLargeError");

  py::class_<pvr::Engine>(module, "Engine")
      .def(py::init([](const py::dict& config) {
        return std::make_unique<pvr::Engine>(parse_config(config));
      }))
      .def("submit",
           [](pvr::Engine& engine, const py::bytes& payload, int media_type) {
             auto copied = copy_bytes(payload);
             py::gil_scoped_release release;
             return engine.submit(std::move(copied),
                                  parse_media_type(media_type));
           })
      .def("submit_many",
           [](pvr::Engine& engine, const py::list& payloads,
              const std::vector<int>& media_types) {
             std::vector<std::vector<std::uint8_t>> copied;
             copied.reserve(payloads.size());
             for (const auto& payload : payloads) {
               copied.push_back(copy_bytes(py::cast<py::bytes>(payload)));
             }
             std::vector<pvr::MediaType> parsed;
             parsed.reserve(media_types.size());
             for (const int value : media_types) {
               parsed.push_back(parse_media_type(value));
             }
             py::gil_scoped_release release;
             return engine.submit_many(std::move(copied), parsed);
           })
       .def("get", [](pvr::Engine& engine, const std::string& id) -> py::object {
         auto record = engine.get(id);
         if (!record) {
           return py::none();
         }
         return record_to_dict(*record);
       })
      .def("get_many", [](pvr::Engine& engine,
                          const std::vector<std::string>& ids) {
        py::list output;
        for (const auto& record : engine.get_many(ids)) {
          output.append(record ? py::object(record_to_dict(*record)) : py::none());
        }
        return output;
      })
      .def("health", [](pvr::Engine& engine) {
        py::dict output;
        output["ready"] = engine.ready();
        output["status"] = engine.ready() ? "ready" : "initializing";
        output["init_error"] = engine.init_error();
        output["queue"] = py::dict("images"_a = engine.queue_images(),
                                   "bytes"_a = engine.queue_bytes());
        output["result_cache"] =
            py::dict("bytes"_a = engine.result_bytes(),
                     "records"_a = engine.result_records());
        const auto [gpu_free, gpu_total] = engine.gpu_memory();
        output["gpu_memory"] =
            py::dict("free_bytes"_a = gpu_free, "total_bytes"_a = gpu_total);
        for (const auto& [key, value] : engine.pipeline_health()) {
          output[py::str(key)] = value;
        }
        return output;
      })
      .def("prometheus", &pvr::Engine::prometheus)
      .def("close", &pvr::Engine::close);
}
