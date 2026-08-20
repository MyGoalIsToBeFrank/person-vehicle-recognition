#pragma once

#include "pvr/types.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pvr {

class Pipeline {
 public:
  explicit Pipeline(const Config& config);
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  std::vector<PipelineResult> run(const std::vector<Task>& tasks);
  std::unordered_map<std::string, std::string> health() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pvr
