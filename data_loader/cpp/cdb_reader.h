#pragma once

#ifdef WITH_CDB

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cdb {

using ScoredMoves = std::vector<std::pair<std::string, int>>;

// Opaque handle to an opened cdb database.
struct CDBHandle;
using Handle = std::shared_ptr<CDBHandle>;

// Open a local cdb dump at `path`.
Handle open(const std::string& path);

// Estimate number of keys.
std::uint64_t size(Handle handle);

// Iterate one rank-local partition of the database.
// `callback` receives (fen, scored_moves); returning false stops this thread's range.
// Ranges are derived from live SST files; contiguous ranges are assigned per rank,
// then split across `num_threads`.
void apply(
    Handle handle,
    std::size_t num_threads,
    std::size_t rank,
    std::size_t world_size,
    const std::function<bool(const std::string& fen, const ScoredMoves& scored)>& callback);

} // namespace cdb

#endif // WITH_CDB
