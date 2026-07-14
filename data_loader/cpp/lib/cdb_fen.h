/*

   The code in the corresponding cdb_fen.cpp file has been authored by
   noobpwnftw and is covered by a BSD-style license (ssdb) and the unlicense
   (chessdb). It was adapted from cdbdirect/fen2cdb.cpp.

   See:
   https://github.com/noobpwnftw/ssdb/blob/124302059809644acc6726a72e234bdb275c56af/src/ssdb/t_hash.h
   https://github.com/noobpwnftw/chessdb/blob/7995300ce3bb68683e6da76e45a94f4c85735d69/extensions/PHP5/cboard/cboard.cpp

*/

#pragma once

#ifdef WITH_CDB

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cdb {

using Bytes = std::string;
using StrPair = std::pair<std::string, std::string>;

std::string cbfen2hexfen(const std::string& fen);
std::string cbhexfen2fen(const std::string& hexfen);
std::string canonicalize_fen_board(const std::string& board);
std::string hex2bin(const std::string& hex);
std::string bin2hex(const std::string& bin);
std::string cbgetBWfen(const std::string& orig);
std::string cbgetBWmove(const std::string& move);
int get_hash_values(const Bytes& slice, std::vector<StrPair>& values);

} // namespace cdb

#endif // WITH_CDB
