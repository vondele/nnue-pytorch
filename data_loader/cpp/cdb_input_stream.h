#pragma once

#ifdef WITH_CDB

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "cdb_reader.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/thread_safe_types.h"

namespace cdb {

using namespace training_data;

struct CDBInputStream : BasicSfenInputStream {
  using SkipPredicate = std::function<bool(const TrainingDataEntry&)>;

  CDBInputStream(
      const std::string& db_path,
      int concurrency,
      bool cyclic,
      int rank,
      int world_size,
      SkipPredicate skipPredicate = nullptr);

  ~CDBInputStream() override;

  std::optional<TrainingDataEntry> next() override;

  void stop() override;

  bool eof() const override;

  static constexpr std::size_t shuffle_buffer_size = 1 << 18;

  // Shuffle + push a buffer to the ring buffer.  Thread-safe.
  void flush_and_put(std::vector<TrainingDataEntry>& buffer);

private:
  static constexpr std::size_t ring_buffer_capacity = 16;

  Handle m_handle;
  int m_concurrency;
  bool m_cyclic;
  int m_rank;
  int m_world_size;
  SkipPredicate m_skipPredicate;

  std::atomic<bool> m_stopFlag{false};
  std::atomic<bool> m_eof{false};

  using RingBuffer = thread_safe_types::ThreadSafeRingBuffer<std::vector<TrainingDataEntry>, ring_buffer_capacity>;
  std::unique_ptr<RingBuffer> m_ringBuffer;
  std::vector<std::thread> m_workers;

  std::vector<TrainingDataEntry> m_localBuffer;
  std::size_t m_localOffset = 0;

  void start_workers();
  std::optional<TrainingDataEntry> convert(
      const std::string& fen, const ScoredMoves& scored);
};

} // namespace cdb

#endif // WITH_CDB
