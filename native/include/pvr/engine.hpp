#pragma once

#include "pvr/pipeline.hpp"
#include "pvr/types.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pvr {

class QueueFullError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class NotReadyError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class PayloadTooLargeError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Engine {
 public:
  explicit Engine(Config config);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  std::string submit(std::vector<std::uint8_t> payload, MediaType media_type);
  std::vector<std::string> submit_many(
      std::vector<std::vector<std::uint8_t>> payloads,
      const std::vector<MediaType>& media_types);
  std::optional<Record> get(const std::string& id);
  std::vector<std::optional<Record>> get_many(
      const std::vector<std::string>& ids);
  std::unordered_map<std::string, std::string> pipeline_health() const;
  bool ready() const noexcept { return ready_.load(std::memory_order_acquire); }
  std::string init_error() const;
  std::size_t queue_images() const;
  std::size_t queue_bytes() const;
  std::size_t result_bytes() const;
  std::size_t result_records() const;
  std::pair<std::size_t, std::size_t> gpu_memory() const;
  std::string prometheus() const;
  void close();

 private:
  struct Metrics {
    std::atomic<std::uint64_t> accepted{0};
    std::atomic<std::uint64_t> completed{0};
    std::atomic<std::uint64_t> rejected{0};
    std::atomic<std::uint64_t> errors{0};
    std::atomic<std::uint64_t> batches{0};
    std::atomic<std::uint64_t> batch_images{0};
    std::atomic<std::uint64_t> latency_sum_us{0};
    std::array<std::atomic<std::uint64_t>, 8> latency_buckets{};
  };

  void worker_loop();
  std::vector<Task> take_batch(std::unique_lock<std::mutex>& lock);
  void store_batch(const std::vector<Task>& tasks,
                   std::vector<PipelineResult> outputs,
                   double inference_ms);
  void expire_locked(std::chrono::steady_clock::time_point now);
  std::string next_id();

  Config config_;
  mutable std::mutex mutex_;
  std::condition_variable available_;
  std::deque<Task> queue_;
  std::unordered_map<std::string, Record> records_;
  std::deque<std::string> completed_order_;
  std::deque<std::string> expired_order_;
  std::size_t queued_bytes_ = 0;
  std::size_t result_bytes_ = 0;
  std::unique_ptr<Pipeline> pipeline_;
  std::thread worker_;
  std::atomic<bool> ready_{false};
  std::atomic<bool> stopping_{false};
  std::atomic<std::uint64_t> id_counter_{0};
  mutable std::mutex init_mutex_;
  std::string init_error_;
  Metrics metrics_;
};

const char* status_name(Status status);

}  // namespace pvr
