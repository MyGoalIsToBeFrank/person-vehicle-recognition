#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pvr {

enum class MediaType : std::uint8_t { jpeg = 1, png = 2, bmp = 3, webp = 4 };
enum class Status { pending, running, done, error, expired };

struct Config {
  std::string model_dir;
  std::string engine_cache_dir;
  std::size_t max_image_bytes = 8ULL << 20;
  std::size_t max_image_pixels = 20'000'000;
  std::size_t max_batch_images = 64;
  std::size_t max_queue_images = 8192;
  std::size_t max_queue_bytes = 1ULL << 30;
  std::chrono::microseconds batch_wait{2000};
  std::chrono::seconds result_ttl{60};
  std::size_t max_result_bytes = 1ULL << 30;
  std::size_t max_result_records = 262'144;
};

struct Task {
  std::string id;
  std::shared_ptr<std::vector<std::uint8_t>> payload;
  MediaType media_type;
  std::chrono::steady_clock::time_point accepted_at;
};

struct PipelineResult {
  bool ok = false;
  std::string json;
  std::string error;
};

struct Record {
  Status status = Status::pending;
  std::string result_json;
  std::string error;
  double queue_ms = 0.0;
  double inference_ms = 0.0;
  double total_ms = 0.0;
  std::size_t retained_bytes = 0;
  std::chrono::steady_clock::time_point accepted_at;
  std::chrono::steady_clock::time_point completed_at;
  std::chrono::steady_clock::time_point expired_at;
};

}  // namespace pvr
