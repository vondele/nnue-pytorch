#ifdef WITH_CDB

#include "cdb_input_stream.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "lib/chess.h"
#include "lib/rng.h"
#include "lib/training_data_entry.h"

namespace cdb {

using namespace training_data;

namespace {

// Per-reader-thread buffer.  Declared thread_local inside the cdb::apply
// callback so each reader thread gets its own instance.  The destructor flushes
// any remaining entries when the thread finishes its range.
struct ThreadLocalBuffer {
  std::vector<TrainingDataEntry> entries;
  CDBInputStream* stream;

  explicit ThreadLocalBuffer(CDBInputStream* s) : stream(s) {
    entries.reserve(CDBInputStream::shuffle_buffer_size);
  }

  ~ThreadLocalBuffer() {
    if (stream) {
      stream->flush_and_put(entries);
    }
  }
};

int sample_result_from_wdl(double w, double d, double l) {
  std::uniform_real_distribution<double> dist(0.0, w + d + l);
  double u = dist(rng::get_thread_local_rng());
  if (u < w) return 1;
  if (u < w + d) return 0;
  return -1;
}

} // namespace

CDBInputStream::CDBInputStream(
    const std::string& db_path,
    int concurrency,
    bool cyclic,
    int rank,
    int world_size,
    SkipPredicate skipPredicate)
    : m_handle(open(db_path)),
      m_concurrency(concurrency),
      m_cyclic(cyclic),
      m_rank(rank),
      m_world_size(world_size),
      m_skipPredicate(std::move(skipPredicate)) {
  if (!m_handle) {
    throw std::runtime_error("Failed to open cdb: " + db_path);
  }
  if (m_concurrency < 1) m_concurrency = 1;
  if (m_rank < 0) m_rank = 0;
  if (m_world_size < 1) m_world_size = 1;

  m_ringBuffer = std::make_unique<RingBuffer>();
  m_ringBuffer->reserve_internal(shuffle_buffer_size);

  start_workers();
}

void CDBInputStream::stop() {
  m_stopFlag.store(true);
  if (m_ringBuffer) {
    m_ringBuffer->signal_stop();
  }
}

CDBInputStream::~CDBInputStream() {
  stop();
  for (auto& t : m_workers) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void CDBInputStream::flush_and_put(std::vector<TrainingDataEntry>& buffer) {
  if (buffer.empty() || m_stopFlag.load()) {
    buffer.clear();
    return;
  }

  auto& prng = rng::get_thread_local_rng();
  std::shuffle(buffer.begin(), buffer.end(), prng);
  m_ringBuffer->put(buffer, [this]() { return m_stopFlag.load(); });
  buffer.clear();
}

void CDBInputStream::start_workers() {
  auto worker = [this]() {
    try {
      do {
        // cdb::apply spawns m_concurrency internal threads and partitions the
        // rank-local range across them.  Each internal thread keeps its own
        // thread_local shuffle buffer (see ThreadLocalBuffer above).
        cdb::apply(m_handle, m_concurrency, m_rank, m_world_size,
                   [&](const std::string& fen, const ScoredMoves& scored) -> bool {
                     if (m_stopFlag.load()) return false;

                     thread_local ThreadLocalBuffer tls(this);
                     auto& buffer = tls.entries;

                     auto entry_opt = convert(fen, scored);
                     if (!entry_opt) return true;

                     buffer.push_back(std::move(*entry_opt));

                     if (buffer.size() >= shuffle_buffer_size) {
                       flush_and_put(buffer);
                       buffer.reserve(shuffle_buffer_size);
                     }

                     return true;
                   });
      } while (m_cyclic && !m_stopFlag.load());
    } catch (const std::exception& e) {
      std::cerr << "CDBInputStream worker exception: " << e.what() << std::endl;
    } catch (...) {
      std::cerr << "CDBInputStream worker unknown exception" << std::endl;
    }
    m_stopFlag.store(true);
    if (m_ringBuffer) {
      m_ringBuffer->signal_stop();
    }
  };

  m_workers.emplace_back(worker);
}

std::optional<TrainingDataEntry> CDBInputStream::convert(
    const std::string& fen, const ScoredMoves& scored) {
  if (scored.empty()) return std::nullopt;

  auto pos_opt = chess::Position::tryFromFen(fen);
  if (!pos_opt) return std::nullopt;

  auto& pos = *pos_opt;

  // Best move from cdb is the first scored move.
  const std::string& best_move_str = scored[0].first;
  if (best_move_str.empty() || best_move_str == "a0a0") return std::nullopt;

  chess::Move best_move;
  try {
    best_move = chess::uci::uciToMove(pos, best_move_str);
  } catch (...) {
    return std::nullopt;
  }

  // Best-move filter: skip positions in check, illegal moves, capturing moves,
  // or moves giving check.
  if (pos.isCheck()) return std::nullopt;
  if (!pos.isMoveLegal(best_move)) return std::nullopt;
  if (pos.pieceAt(best_move.to) != chess::Piece::none()) return std::nullopt;
  if (pos.isCheckAfterMove(best_move)) return std::nullopt;

  TrainingDataEntry entry;
  entry.pos = pos;
  entry.move = best_move;

  // Ply from the sentinel a0a0 entry; default to 0 if unknown.
  int ply = scored.back().second;
  if (ply < 0) ply = 0;
  entry.ply = static_cast<std::uint16_t>(ply);
  entry.pos.setPly(entry.ply);

  // Score conversion. cdb scores are centipawns; the internal training score
  // unit is such that centipawns = 100 * score / 208, so convert with
  // score = cdb_score * 208 / 100.
  int cdb_score = scored[0].second;
  if (cdb_score == -30001) {
    // TB draw / stalemate.
    entry.score = 0;
  } else if (cdb_score >= 30000) {
    entry.score = static_cast<std::int16_t>(32000 - ply);
  } else if (cdb_score <= -30000) {
    entry.score = static_cast<std::int16_t>(-(32000 - ply));
  } else if (cdb_score >= 15000) {
    entry.score = static_cast<std::int16_t>(31000 - ply);
  } else if (cdb_score <= -15000) {
    entry.score = static_cast<std::int16_t>(-(31000 - ply));
  } else {
    entry.score = static_cast<std::int16_t>(std::clamp(
        static_cast<int>(std::llround(cdb_score * 208.0 / 100.0)),
        static_cast<int>(std::numeric_limits<std::int16_t>::min()),
        static_cast<int>(std::numeric_limits<std::int16_t>::max())));
  }

  // Result: sample from WDL probabilities derived from score and ply.
  // win_rate_model() returns (win, loss, draw).
  auto [w, l, d] = entry.win_rate_model();
  entry.result = sample_result_from_wdl(w, l, d);

  if (m_skipPredicate && !m_skipPredicate(entry)) {
    return std::nullopt;
  }

  return entry;
}

std::optional<TrainingDataEntry> CDBInputStream::next() {
  while (true) {
    if (m_localOffset < m_localBuffer.size()) {
      return std::move(m_localBuffer[m_localOffset++]);
    }

    m_localBuffer.clear();
    m_localOffset = 0;

    if (!m_ringBuffer->take(m_localBuffer, [this]() { return m_stopFlag.load() || m_eof.load(); })) {
      m_eof.store(true);
      return std::nullopt;
    }
  }
}

bool CDBInputStream::eof() const {
  return m_eof.load();
}

} // namespace cdb

#endif // WITH_CDB
