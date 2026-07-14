#ifdef WITH_CDB

#include "cdb_reader.h"
#include "lib/cdb_fen.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "table/terark_zip_table.h"

namespace cdb {

using namespace TERARKDB_NAMESPACE;

namespace {

// Parse a signed decimal integer from a DB value string. Returns true on success.
bool safe_parse_int(const std::string& s, int& out) {
  if (s.empty()) {
    return false;
  }
  const char* p = s.data();
  std::size_t n = s.size();
  std::size_t i = 0;
  bool neg = false;
  if (p[0] == '-') {
    neg = true;
    i = 1;
  } else if (p[0] == '+') {
    i = 1;
  }
  if (i >= n) {
    return false;
  }
  unsigned long long val = 0;
  constexpr unsigned long long kIntMax = static_cast<unsigned long long>(std::numeric_limits<int>::max());
  constexpr unsigned long long kIntMinAbs = static_cast<unsigned long long>(std::numeric_limits<int>::max()) + 1;
  const unsigned long long limit = neg ? kIntMinAbs : kIntMax;
  for (; i < n; ++i) {
    char c = p[i];
    if (c < '0' || c > '9') {
      return false;
    }
    val = val * 10 + static_cast<unsigned>(c - '0');
    if (val > limit) {
      return false;
    }
  }
  out = neg ? -static_cast<int>(val) : static_cast<int>(val);
  return true;
}

} // namespace

enum class STM { WHITE, BLACK, NONE };

STM fen_to_stm(const std::string& fen) {
  return fen.find(" w ") != std::string::npos ? STM::WHITE : STM::BLACK;
}

enum class MinPlyType { SINGLE, DUAL, NONE };

struct CDBHandle {
  DB* db = nullptr;
  MinPlyType min_ply_type = MinPlyType::NONE;

  ~CDBHandle() {
    if (db != nullptr) {
      delete db;
      db = nullptr;
    }
  }
};

// scores outside of [-15000, 15000] are assumed to be (cursed) TB wins or
// (possibly incorrect) mates, apart from TB draws and stalemates, which are
// both encoded with -30001
int backprop_score(int child_score) {
  if (child_score == -30001)
    return 0;
  if (child_score >= 15000)
    return -child_score + 1;
  if (child_score <= -15000)
    return -child_score - 1;
  return -child_score;
}

std::string canonicalize_fen(const std::string& raw_fen) {
  // The raw FEN from cbhexfen2fen uses individual '1' chars for each empty square,
  // but the chess library expects compressed runs (e.g. "8" instead of "11111111").
  std::size_t space = raw_fen.find(' ');
  if (space == std::string::npos) return raw_fen;
  std::string board = raw_fen.substr(0, space);
  return canonicalize_fen_board(board) + raw_fen.substr(space);
}

std::vector<std::pair<std::string, int>> value_to_scoredMoves(
    const std::string& value, STM key_stm, STM& fen_stm, MinPlyType min_ply_type) {

  if (value.empty()) {
    return {{"a0a0", -2}};
  }

  std::vector<StrPair> scoredMoves;
  get_hash_values(value, scoredMoves);

  std::vector<std::pair<std::string, int>> result;
  result.reserve(2);

  int white_ply = -1, black_ply = -1;
  std::string best_move;
  int best_score = std::numeric_limits<int>::min();
  for (auto& pair : scoredMoves) {
    if (pair.first == "a0a0") {
      int ply = -1;
      if (!safe_parse_int(pair.second, ply)) {
        continue;
      }
      switch (min_ply_type) {
      case MinPlyType::SINGLE:
        ply = std::max(ply, -1);
        if (ply >= 0) {
          white_ply = ply % 2 ? ply + 1 : ply;
          black_ply = ply % 2 ? ply : ply + 1;
        }
        break;

      case MinPlyType::DUAL:
        white_ply = std::max(2 * (ply >> 8) - 2, -1);
        black_ply = 2 * (ply & 0xFF) - 1;
        break;

      case MinPlyType::NONE:
        break;
      }
    } else {
      int score = 0;
      if (!safe_parse_int(pair.second, score)) {
        continue;
      }
      int propagated = backprop_score(score);
      if (propagated > best_score) {
        best_score = propagated;
        best_move = pair.first;
      }
    }
  }

  if (fen_stm == STM::NONE) {
    if (white_ply >= 0 && (black_ply < 0 || white_ply < black_ply))
      fen_stm = STM::WHITE;
    else if (black_ply >= 0 && (white_ply < 0 || white_ply > black_ply))
      fen_stm = STM::BLACK;
    else
      fen_stm = key_stm;
  }

  if (!best_move.empty()) {
    result.push_back({best_move, best_score});
  }

  result.push_back({"a0a0", fen_stm == STM::WHITE ? white_ply : black_ply});

  return result;
}

MinPlyType detect_min_ply_type(DB* db) {
  ReadOptions read_options;
  read_options.verify_checksums = false;
  std::string value;
  auto startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";
  auto hexfen = cbfen2hexfen(startpos);
  auto BWfen = cbgetBWfen(startpos);
  auto BWhexfen = cbfen2hexfen(BWfen);
  std::string key = 'h' + hex2bin(std::min(hexfen, BWhexfen));
  Status s = db->Get(read_options, key, &value);
  if (!s.ok()) {
    std::cerr << "Could not probe startpos for min_ply detection: " << s.ToString() << std::endl;
    return MinPlyType::NONE;
  }

  std::vector<StrPair> scoredMoves;
  get_hash_values(value, scoredMoves);
  int ply = -1;
  for (auto& pair : scoredMoves) {
    if (pair.first == "a0a0") {
      if (!safe_parse_int(pair.second, ply)) {
        ply = -1;
      }
      break;
    }
  }

  switch (ply) {
  case 0:
    return MinPlyType::SINGLE;
  case 256:
    return MinPlyType::DUAL;
  default:
    std::cerr << "Could not detect min_ply encoding scheme, ply = " << ply << std::endl;
    return MinPlyType::NONE;
  }
}

Handle open(const std::string& path) {
  TerarkZipTableOptions tzt_options;
  tzt_options.localTempDir = "/tmp";
  tzt_options.warmUpIndexOnOpen = false;
  tzt_options.minPreadLen = 0;
  tzt_options.indexCacheRatio = 0.000;
  tzt_options.cacheCapacityBytes = 1 * 1024 * 1024 * 1024LL;

  Options options;
  options.IncreaseParallelism();
  options.table_factory.reset(NewTerarkZipTableFactory(tzt_options, options.table_factory));

  auto handle = std::make_shared<CDBHandle>();
  Status s = DB::OpenForReadOnly(options, path, &handle->db);
  if (!s.ok()) {
    std::cerr << s.ToString() << std::endl;
    return nullptr;
  }

  handle->min_ply_type = detect_min_ply_type(handle->db);
  return handle;
}

std::uint64_t size(Handle handle) {
  std::uint64_t value = 0;
  if (!handle->db->GetIntProperty("rocksdb.estimate-num-keys", &value)) {
    return 0;
  }
  return value;
}

struct RangeStorage {
  std::string start;
  std::string limit;
  RangeStorage() = default;
  RangeStorage(const std::string& s, const std::string& l) : start(s), limit(l) {}
};

std::vector<RangeStorage> buildMergedRanges(DB* db) {
  ReadOptions read_options;
  read_options.verify_checksums = false;
  std::unique_ptr<Iterator> it(db->NewIterator(read_options));
  it->SeekToLast();
  if (!it->Valid()) {
    return {};
  }
  std::string last_key_str = it->key().ToString();

  std::vector<LiveFileMetaData> files;
  db->GetLiveFilesMetaData(&files);
  const Comparator* cmp = db->GetOptions().comparator;

  std::sort(files.begin(), files.end(),
            [&cmp](const LiveFileMetaData& a, const LiveFileMetaData& b) {
              return cmp->Compare(a.smallestkey, b.smallestkey) < 0;
            });

  std::vector<RangeStorage> merged;
  for (size_t i = 0; i < files.size(); ++i) {
    merged.push_back(RangeStorage(
        files[i].smallestkey,
        (i + 1 == files.size()) ? last_key_str + '\xff' : files[i + 1].smallestkey));
  }
  return merged;
}

std::vector<RangeStorage> buildDDPThreadRanges(
    DB* db, std::size_t num_threads, std::size_t rank, std::size_t world_size) {
  auto merged = buildMergedRanges(db);

  // Assign contiguous merged ranges to ranks.
  std::size_t num_rank_ranges = std::max<std::size_t>(1, merged.size() / world_size);
  std::size_t rank_start = std::min(rank * num_rank_ranges, merged.size());
  std::size_t rank_end = (rank == world_size - 1)
                             ? merged.size()
                             : std::min((rank + 1) * num_rank_ranges, merged.size());

  std::vector<RangeStorage> rank_ranges(merged.begin() + rank_start, merged.begin() + rank_end);

  // Split rank-local ranges across threads.
  std::size_t num_threads_eff = std::min(num_threads, rank_ranges.size());
  std::size_t per_thread = rank_ranges.size() / num_threads_eff;
  std::size_t remainder = rank_ranges.size() % num_threads_eff;

  std::vector<RangeStorage> out;
  std::size_t idx = 0;
  for (std::size_t i = 0; i < num_threads_eff; ++i) {
    std::size_t chunk = per_thread + (i < remainder ? 1 : 0);
    if (chunk == 0) break;
    out.push_back(RangeStorage(rank_ranges[idx].start, rank_ranges[idx + chunk - 1].limit));
    idx += chunk;
  }
  return out;
}

void iterateRange(
    CDBHandle* handle, const RangeStorage& range,
    const std::function<bool(const std::string&, const ScoredMoves&)>& callback) {
  const Comparator* cmp = handle->db->GetOptions().comparator;
  ReadOptions read_options;
  read_options.verify_checksums = false;
  std::unique_ptr<Iterator> it(handle->db->NewIterator(read_options));

  for (it->Seek(range.start);
       it->Valid() && (cmp->Compare(it->key(), range.limit) < 0);
       it->Next()) {
    // Decode only the canonical FEN for the key's side-to-move up front.
    // The black-to-move FEN is computed lazily only when the ply encoding
    // tells us the position belongs to the opposite side.
    auto key_str = it->key().ToString();
    auto raw_fen = canonicalize_fen(cbhexfen2fen(bin2hex(key_str.substr(1))));
    STM key_stm = fen_to_stm(raw_fen);
    STM fen_stm = STM::NONE;
    auto scored = value_to_scoredMoves(it->value().ToString(), key_stm, fen_stm, handle->min_ply_type);

    const std::string& fen = (key_stm == fen_stm) ? raw_fen : cbgetBWfen(raw_fen);
    if (key_stm != fen_stm && !scored.empty() && scored[0].first != "a0a0") {
      scored[0].first = cbgetBWmove(scored[0].first);
    }

    if (!callback(fen, scored))
      break;
  }
}

void apply(
    Handle handle,
    std::size_t num_threads,
    std::size_t rank,
    std::size_t world_size,
    const std::function<bool(const std::string& fen, const ScoredMoves& scored)>& callback) {
  auto ranges = buildDDPThreadRanges(handle->db, num_threads, rank, world_size);

  std::vector<std::thread> workers;
  for (auto& r : ranges) {
    workers.emplace_back(iterateRange, handle.get(), r, callback);
  }
  for (auto& t : workers)
    t.join();
}

} // namespace cdb

#endif // WITH_CDB
