#include "pvr/engine.hpp"

#include "pvr/cuda_utils.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>
#include <utility>

namespace pvr {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

constexpr std::array<double, 8> kLatencyBoundsMs = {
    10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0};

}  // namespace

const char* status_name(Status status) {
  switch (status) {
    case Status::pending:
      return "pending";
    case Status::running:
      return "running";
    case Status::done:
      return "done";
    case Status::error:
      return "error";
    case Status::expired:
      return "expired";
  }
  return "error";
}

Engine::Engine(Config config) : config_(std::move(config)) {
  if (config_.max_batch_images == 0 ||
      config_.max_queue_images < config_.max_batch_images ||
      config_.max_result_records < config_.max_queue_images ||
      config_.max_image_bytes == 0 || config_.max_queue_bytes == 0 ||
      config_.max_result_bytes == 0) {
    throw std::invalid_argument("invalid native engine capacity limits");
  }
  worker_ = std::thread(&Engine::worker_loop, this);
}

Engine::~Engine() { close(); }

void Engine::close() {
  if (stopping_.exchange(true)) {
    return;
  }
  available_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

std::string Engine::next_id() {
  static thread_local std::mt19937_64 random{std::random_device{}()};
  const auto sequence = id_counter_.fetch_add(1, std::memory_order_relaxed);
  const auto high = random();
  const auto low = random() ^ sequence;
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << high
         << std::setw(16) << low;
  return output.str();
}

std::string Engine::submit(std::vector<std::uint8_t> payload,
                           MediaType media_type) {
  auto ids = submit_many({std::move(payload)}, {media_type});
  return std::move(ids.front());
}

std::vector<std::string> Engine::submit_many(
    std::vector<std::vector<std::uint8_t>> payloads,
    const std::vector<MediaType>& media_types) {
  if (!ready()) {
    throw NotReadyError(init_error().empty() ? "engine is initializing"
                                             : init_error());
  }
  if (payloads.empty() || payloads.size() != media_types.size() ||
      payloads.size() > config_.max_batch_images) {
    throw PayloadTooLargeError("invalid submit batch size");
  }
  std::size_t incoming_bytes = 0;
  for (const auto& payload : payloads) {
    if (payload.empty() || payload.size() > config_.max_image_bytes) {
      throw PayloadTooLargeError("image byte limit exceeded");
    }
    incoming_bytes += payload.size();
  }

  std::vector<std::string> ids;
  ids.reserve(payloads.size());
  const auto now = Clock::now();
  std::lock_guard lock(mutex_);
  expire_locked(now);
  while (records_.size() + payloads.size() > config_.max_result_records &&
         !expired_order_.empty()) {
    const auto found = records_.find(expired_order_.front());
    if (found != records_.end() && found->second.status == Status::expired) {
      records_.erase(found);
    }
    expired_order_.pop_front();
  }
  if (queue_.size() + payloads.size() > config_.max_queue_images ||
      queued_bytes_ + incoming_bytes > config_.max_queue_bytes ||
      records_.size() + payloads.size() > config_.max_result_records) {
    metrics_.rejected.fetch_add(payloads.size(), std::memory_order_relaxed);
    throw QueueFullError("bounded native queue is full");
  }

  for (std::size_t i = 0; i < payloads.size(); ++i) {
    auto id = next_id();
    auto bytes = std::make_shared<std::vector<std::uint8_t>>(
        std::move(payloads[i]));
    queued_bytes_ += bytes->size();
    records_.emplace(id, Record{.accepted_at = now});
    queue_.push_back(Task{.id = id,
                          .payload = std::move(bytes),
                          .media_type = media_types[i],
                          .accepted_at = now});
    ids.push_back(std::move(id));
  }
  metrics_.accepted.fetch_add(payloads.size(), std::memory_order_relaxed);
  available_.notify_one();
  return ids;
}

std::optional<Record> Engine::get(const std::string& id) {
  std::lock_guard lock(mutex_);
  expire_locked(Clock::now());
  const auto found = records_.find(id);
  if (found == records_.end()) {
    return std::nullopt;
  }
  return found->second;
}

std::vector<std::optional<Record>> Engine::get_many(
    const std::vector<std::string>& ids) {
  std::vector<std::optional<Record>> output;
  output.reserve(ids.size());
  std::lock_guard lock(mutex_);
  expire_locked(Clock::now());
  for (const auto& id : ids) {
    const auto found = records_.find(id);
    output.push_back(found == records_.end() ? std::nullopt
                                             : std::optional(found->second));
  }
  return output;
}

std::vector<Task> Engine::take_batch(std::unique_lock<std::mutex>& lock) {
  available_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
  if (stopping_) {
    return {};
  }
  const auto deadline = Clock::now() + config_.batch_wait;
  available_.wait_until(lock, deadline, [this] {
    return stopping_ || queue_.size() >= config_.max_batch_images;
  });
  if (stopping_) {
    return {};
  }

  const auto count = std::min(queue_.size(), config_.max_batch_images);
  std::vector<Task> batch;
  batch.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    Task task = std::move(queue_.front());
    queue_.pop_front();
    queued_bytes_ -= task.payload->size();
    auto record = records_.find(task.id);
    if (record != records_.end()) {
      record->second.status = Status::running;
      record->second.queue_ms = milliseconds(Clock::now() - task.accepted_at);
    }
    batch.push_back(std::move(task));
  }
  return batch;
}

void Engine::store_batch(const std::vector<Task>& tasks,
                         std::vector<PipelineResult> outputs,
                         double inference_ms) {
  const auto now = Clock::now();
  std::lock_guard lock(mutex_);
  if (outputs.size() != tasks.size()) {
    outputs.assign(tasks.size(),
                   PipelineResult{.ok = false,
                                  .error = "native pipeline result count mismatch"});
  }
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    auto found = records_.find(tasks[i].id);
    if (found == records_.end()) {
      continue;
    }
    auto& record = found->second;
    record.completed_at = now;
    record.inference_ms = inference_ms;
    record.total_ms = milliseconds(now - record.accepted_at);
    if (outputs[i].ok) {
      record.status = Status::done;
      record.result_json = std::move(outputs[i].json);
      record.retained_bytes = record.result_json.size();
    } else {
      record.status = Status::error;
      record.error = std::move(outputs[i].error);
      record.retained_bytes = record.error.size();
      metrics_.errors.fetch_add(1, std::memory_order_relaxed);
    }
    result_bytes_ += record.retained_bytes;
    completed_order_.push_back(tasks[i].id);
    metrics_.completed.fetch_add(1, std::memory_order_relaxed);
    metrics_.latency_sum_us.fetch_add(
        static_cast<std::uint64_t>(record.total_ms * 1000.0),
        std::memory_order_relaxed);
    for (std::size_t bucket = 0; bucket < kLatencyBoundsMs.size(); ++bucket) {
      if (record.total_ms <= kLatencyBoundsMs[bucket]) {
        metrics_.latency_buckets[bucket].fetch_add(1,
                                                   std::memory_order_relaxed);
      }
    }
  }
  expire_locked(now);
}

void Engine::expire_locked(Clock::time_point now) {
  while (!expired_order_.empty()) {
    const auto found = records_.find(expired_order_.front());
    if (found == records_.end() || found->second.status != Status::expired) {
      expired_order_.pop_front();
      continue;
    }
    const bool tombstone_ttl_elapsed =
        now - found->second.expired_at >= config_.result_ttl;
    const bool over_record_budget =
        records_.size() >= config_.max_result_records;
    if (!tombstone_ttl_elapsed && !over_record_budget) {
      break;
    }
    records_.erase(found);
    expired_order_.pop_front();
  }

  while (!completed_order_.empty()) {
    const auto found = records_.find(completed_order_.front());
    if (found == records_.end()) {
      completed_order_.pop_front();
      continue;
    }
    auto& record = found->second;
    const bool ttl_elapsed = now - record.completed_at >= config_.result_ttl;
    const bool over_budget = result_bytes_ > config_.max_result_bytes;
    if (!ttl_elapsed && !over_budget) {
      break;
    }
    result_bytes_ -= record.retained_bytes;
    record.retained_bytes = 0;
    record.result_json.clear();
    record.error.clear();
    record.status = Status::expired;
    record.expired_at = now;
    expired_order_.push_back(completed_order_.front());
    completed_order_.pop_front();
  }
}

void Engine::worker_loop() {
  try {
    auto pipeline = std::make_unique<Pipeline>(config_);
    {
      std::lock_guard lock(mutex_);
      pipeline_ = std::move(pipeline);
      ready_.store(true, std::memory_order_release);
    }
  } catch (const std::exception& error) {
    std::lock_guard lock(init_mutex_);
    init_error_ = error.what();
    return;
  }

  while (!stopping_) {
    std::unique_lock lock(mutex_);
    auto tasks = take_batch(lock);
    lock.unlock();
    if (tasks.empty()) {
      continue;
    }
    const auto started = Clock::now();
    std::vector<PipelineResult> output;
    try {
      output = pipeline_->run(tasks);
    } catch (const std::exception& error) {
      output.assign(tasks.size(),
                    PipelineResult{.ok = false, .error = error.what()});
    }
    const double elapsed = milliseconds(Clock::now() - started);
    metrics_.batches.fetch_add(1, std::memory_order_relaxed);
    metrics_.batch_images.fetch_add(tasks.size(), std::memory_order_relaxed);
    store_batch(tasks, std::move(output), elapsed);
  }
}

std::unordered_map<std::string, std::string> Engine::pipeline_health() const {
  std::lock_guard lock(mutex_);
  if (!pipeline_) {
    return {};
  }
  return pipeline_->health();
}

std::string Engine::init_error() const {
  std::lock_guard lock(init_mutex_);
  return init_error_;
}

std::size_t Engine::queue_images() const {
  std::lock_guard lock(mutex_);
  return queue_.size();
}

std::size_t Engine::queue_bytes() const {
  std::lock_guard lock(mutex_);
  return queued_bytes_;
}

std::size_t Engine::result_bytes() const {
  std::lock_guard lock(mutex_);
  return result_bytes_;
}

std::size_t Engine::result_records() const {
  std::lock_guard lock(mutex_);
  return records_.size();
}

std::pair<std::size_t, std::size_t> Engine::gpu_memory() const {
  std::size_t free = 0;
  std::size_t total = 0;
  check_cuda(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
  return {free, total};
}

std::string Engine::prometheus() const {
  const auto [gpu_free, gpu_total] = gpu_memory();
  std::ostringstream output;
  output << "# TYPE pvr_images_accepted_total counter\n"
         << "pvr_images_accepted_total " << metrics_.accepted.load() << "\n"
         << "# TYPE pvr_images_completed_total counter\n"
         << "pvr_images_completed_total " << metrics_.completed.load() << "\n"
         << "# TYPE pvr_images_rejected_total counter\n"
         << "pvr_images_rejected_total " << metrics_.rejected.load() << "\n"
         << "# TYPE pvr_image_errors_total counter\n"
         << "pvr_image_errors_total " << metrics_.errors.load() << "\n"
         << "# TYPE pvr_gpu_batches_total counter\n"
         << "pvr_gpu_batches_total " << metrics_.batches.load() << "\n"
         << "# TYPE pvr_gpu_batch_images_total counter\n"
         << "pvr_gpu_batch_images_total " << metrics_.batch_images.load() << "\n"
         << "# TYPE pvr_queue_images gauge\n"
         << "pvr_queue_images " << queue_images() << "\n"
         << "# TYPE pvr_queue_bytes gauge\n"
         << "pvr_queue_bytes " << queue_bytes() << "\n"
         << "# TYPE pvr_result_bytes gauge\n"
         << "pvr_result_bytes " << result_bytes() << "\n"
         << "# TYPE pvr_result_records gauge\n"
         << "pvr_result_records " << result_records() << "\n"
         << "# TYPE pvr_gpu_memory_free_bytes gauge\n"
         << "pvr_gpu_memory_free_bytes " << gpu_free << "\n"
         << "# TYPE pvr_gpu_memory_total_bytes gauge\n"
         << "pvr_gpu_memory_total_bytes " << gpu_total << "\n"
         << "# TYPE pvr_total_latency_ms histogram\n";
  for (std::size_t i = 0; i < kLatencyBoundsMs.size(); ++i) {
    output << "pvr_total_latency_ms_bucket{le=\"" << kLatencyBoundsMs[i]
           << "\"} " << metrics_.latency_buckets[i].load() << "\n";
  }
  output << "pvr_total_latency_ms_bucket{le=\"+Inf\"} "
         << metrics_.completed.load() << "\n"
         << "pvr_total_latency_ms_sum "
         << metrics_.latency_sum_us.load() / 1000.0 << "\n"
         << "pvr_total_latency_ms_count " << metrics_.completed.load() << "\n";
  return output.str();
}

}  // namespace pvr
