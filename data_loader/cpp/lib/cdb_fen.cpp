/*

   The code in this file has been authored by noobpwnftw and is covered by the
original licenses a BSD-style license (ssdb) and the unlicense (chessdb). See
   https://github.com/noobpwnftw/ssdb/blob/124302059809644acc6726a72e234bdb275c56af/src/ssdb/t_hash.h
   https://github.com/noobpwnftw/chessdb/blob/7995300ce3bb68683e6da76e45a94f4c85735d69/extensions/PHP5/cboard/cboard.cpp

---
Copyright (c) 2013 SSDB Authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the SSDB nor the names of its contributors may be used
to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>

*/

#ifdef WITH_CDB

#include "cdb_fen.h"

#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace cdb {

#define CHESS_FEN_MAX_LENGTH 128
#define CHESS_BITSTR_MAX_LENGTH 93

char char2bithex(char ch) {
  switch (ch) {
  case '1':
    return '0';
  case '2':
    return '1';
  case '3':
    return '2';
  case 'p':
    return '3';
  case 'n':
    return '4';
  case 'b':
    return '5';
  case 'r':
    return '6';
  case 'q':
    return '7';
  case 'k':
    return '9';
  case 'P':
    return 'a';
  case 'N':
    return 'b';
  case 'B':
    return 'c';
  case 'R':
    return 'd';
  case 'Q':
    return 'e';
  case 'K':
    return 'f';
  default:
    return '8';
  }
}

char bithex2char(unsigned char ch) {
  switch (ch) {
  case '0':
    return '1';
  case '1':
    return '2';
  case '2':
    return '3';
  case '3':
    return 'p';
  case '4':
    return 'n';
  case '5':
    return 'b';
  case '6':
    return 'r';
  case '7':
    return 'q';
  case '9':
    return 'k';
  case 'a':
    return 'P';
  case 'b':
    return 'N';
  case 'c':
    return 'B';
  case 'd':
    return 'R';
  case 'e':
    return 'Q';
  case 'f':
    return 'K';
  }
  return 'x';
}

char extra2bithex(char ch) {
  switch (ch) {
  case '-':
    return '0';
  case 'K':
    return 'a';
  case 'Q':
    return 'b';
  case 'k':
    return 'c';
  case 'q':
    return 'd';
  case 'a':
    return '1';
  case 'b':
    return '2';
  case 'c':
    return '3';
  case 'd':
    return '4';
  case 'e':
    return '5';
  case 'f':
    return '6';
  case 'g':
    return '7';
  case 'h':
    return '8';
  case ' ': // sentinel for FRC inner rooks castling rights
    return '9';
  case 'B':
  case 'C':
  case 'D':
  case 'E':
  case 'F':
  case 'G':
    return 'e';
  default:
    return ch;
  }
}

char bithex2extra(unsigned char ch) {
  switch (ch) {
  case '0':
    return '-';
  case 'a':
    return 'K';
  case 'b':
    return 'Q';
  case 'c':
    return 'k';
  case 'd':
    return 'q';
  case '1':
    return 'a';
  case '2':
    return 'b';
  case '3':
    return 'c';
  case '4':
    return 'd';
  case '5':
    return 'e';
  case '6':
    return 'f';
  case '7':
    return 'g';
  case '8':
    return 'h';
  case '9':
    return ' ';
  }
  return 'x';
}

std::string cbfen2hexfen(const std::string& fen) {
  const char* fenstr = fen.data();
  size_t fenstr_len = fen.size();

  char bitstr[CHESS_BITSTR_MAX_LENGTH];
  size_t index = 0;
  size_t tmpindex = 0;
  const size_t max_index = CHESS_BITSTR_MAX_LENGTH - 1;
  while (index < fenstr_len && tmpindex < max_index) {
    char curCh = fenstr[index];
    if (curCh == ' ') {
      if (index + 1 < fenstr_len && fenstr[index + 1] == 'b') {
        bitstr[tmpindex++] = '1';
      } else {
        bitstr[tmpindex++] = '0';
      }
      index += 3;
      while (index < fenstr_len && tmpindex < max_index) {
        bitstr[tmpindex++] = extra2bithex(fenstr[index++]);
        if (tmpindex > 0 && bitstr[tmpindex - 1] == 'e' && tmpindex < max_index) {
          bitstr[tmpindex++] = extra2bithex(tolower(fenstr[index - 1]));
        }
      }
      break;
    } else if (curCh == '/') {
      index++;
    } else {
      bitstr[tmpindex++] = char2bithex(curCh);
      if (curCh >= '4' && curCh <= '8' && tmpindex < max_index) {
        bitstr[tmpindex++] = curCh - 4;
      }
      index++;
    }
  }
  if (tmpindex % 2) {
    if (tmpindex > 0 && bitstr[tmpindex - 1] == '0') {
      bitstr[tmpindex - 1] = '\0';
    } else if (tmpindex < max_index) {
      bitstr[tmpindex++] = '0';
      bitstr[tmpindex] = '\0';
    } else {
      bitstr[max_index] = '\0';
    }
  } else {
    bitstr[tmpindex] = '\0';
  }
  return std::string(bitstr);
}

std::string canonicalize_fen_board(const std::string& board) {
  std::string out;
  out.reserve(board.size());
  for (char c : board) {
    if (c == '1') {
      if (!out.empty() && out.back() >= '1' && out.back() <= '8') {
        out.back() += 1;
      } else {
        out.push_back(c);
      }
    } else {
      out.push_back(c);
    }
  }
  return out;
}

std::string cbhexfen2fen(const std::string& hexfen) {
  const char* fenstr = hexfen.data();
  size_t fenstr_len = hexfen.size();
  size_t index = 0;
  char fen[CHESS_FEN_MAX_LENGTH];
  size_t tmpidx = 0;
  const size_t max_idx = CHESS_FEN_MAX_LENGTH - 1;
  for (int sq = 0; sq < 64 && tmpidx < max_idx; sq++) {
    if (sq != 0 && (sq % 8) == 0) {
      fen[tmpidx++] = '/';
    }
    char tmpch = '0';
    if (index < fenstr_len) {
      tmpch = fenstr[index++];
    }
    if (tmpch == '1') {
      sq += 1;
    } else if (tmpch == '2') {
      sq += 2;
    }
    if (tmpch == '8' && index < fenstr_len && tmpidx < max_idx) {
      tmpch = fenstr[index++];
      fen[tmpidx++] = tmpch + 4;
      sq += tmpch - '0' + 3;
    } else if (tmpidx < max_idx) {
      fen[tmpidx++] = bithex2char(tmpch);
    }
  }
  fen[tmpidx] = '\0';

  if (tmpidx + 3 >= max_idx) {
    return std::string(fen);
  }
  if (index < fenstr_len && fenstr[index++] != '0') {
    strncat(fen, " b ", max_idx - tmpidx);
  } else {
    strncat(fen, " w ", max_idx - tmpidx);
  }
  tmpidx = std::strlen(fen);
  while (index < fenstr_len && tmpidx < max_idx) {
    if (fenstr[index] == 'e') {
      index++;
      if (index < fenstr_len && tmpidx < max_idx) {
        fen[tmpidx++] = toupper(bithex2extra(fenstr[index++]));
      }
    } else if (index < fenstr_len && tmpidx < max_idx) {
      fen[tmpidx++] = bithex2extra(fenstr[index++]);
    }
    if (tmpidx > 0 && fen[tmpidx - 1] == ' ') {
      break;
    }
  }
  if (index < fenstr_len && tmpidx < max_idx) {
    fen[tmpidx++] = bithex2extra(fenstr[index++]);
    if (tmpidx > 0 && fen[tmpidx - 1] != '-' && index < fenstr_len && tmpidx < max_idx)
      fen[tmpidx++] = fenstr[index];
    fen[tmpidx] = '\0';
  } else if (tmpidx < max_idx) {
    fen[tmpidx++] = '-';
    fen[tmpidx] = '\0';
  }
  return std::string(fen);
}

const char MoveToBW[128] = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   '8', '7', '6', '5', '4', '3', '2', '1',
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
};

std::string cbgetBWfen(const std::string& orig) {
  const char* fenstr = orig.data();
  size_t fenstr_len = orig.size();
  size_t index = 0;

  std::string board;
  std::string current_rank;
  while (index < fenstr_len && fenstr[index] != ' ') {
    char c = fenstr[index++];
    if (c == '/') {
      if (!board.empty()) {
        current_rank += '/';
        current_rank += board;
        board = std::move(current_rank);
      } else {
        board = std::move(current_rank);
      }
      current_rank.clear();
    } else {
      if (isupper(c)) {
        current_rank.push_back(tolower(c));
      } else {
        current_rank.push_back(toupper(c));
      }
    }
  }
  if (!current_rank.empty()) {
    if (!board.empty()) {
      current_rank += '/';
      current_rank += board;
      board = std::move(current_rank);
    } else {
      board = std::move(current_rank);
    }
  }

  std::string result = std::move(board);
  if (index >= fenstr_len) {
    return result;
  }

  // Side to move.
  result += (fenstr[index + 1] == 'w') ? " b " : " w ";
  index += 3;

  // Castling rights.
  std::string black_castling;
  std::string white_castling;
  while (index < fenstr_len && fenstr[index] != ' ') {
    char c = fenstr[index++];
    if (isupper(c)) {
      black_castling.push_back(tolower(c));
    } else {
      white_castling.push_back(toupper(c));
    }
  }
  if (white_castling.empty() && black_castling.empty()) {
    result += '-';
  } else {
    result += white_castling;
    result += black_castling;
  }
  result += ' ';
  index++;

  // En passant, halfmove clock, fullmove number (flip rank digits).
  while (index < fenstr_len) {
    char t = MoveToBW[static_cast<unsigned char>(fenstr[index])];
    result.push_back(t ? t : fenstr[index]);
    index++;
  }

  return result;
}

std::string cbgetBWmove(const std::string& move) {
  if (move.size() < 4 || move.size() > 5) {
    return {};
  }
  std::string BWmove;
  BWmove.reserve(move.size());
  for (size_t i = 0; i < 4; ++i) {
    char t = MoveToBW[static_cast<unsigned char>(move[i])];
    BWmove.push_back(t ? t : move[i]);
  }
  if (move.size() == 5) {
    BWmove.push_back(move[4]);
  }
  return BWmove;
}

std::string hex2bin(const std::string& hex) {
  std::string bin;
  bin.reserve(hex.size() / 2);

  for (size_t i = 0; i < hex.size(); i += 2) {
    char byte = (char)strtol(hex.substr(i, 2).c_str(), NULL, 16);
    bin.push_back(byte);
  }

  return bin;
}

std::string bin2hex(const std::string& bin) {
  std::stringstream ss;

  for (unsigned char c : bin) {
    ss << std::hex << std::setw(2) << std::setfill('0') << (int)c;
  }

  return ss.str();
}

const char SQ_File[90] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'a', 'b', 'c', 'd', 'e', 'f',
    'g', 'h', 'i', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'a', 'b', 'c',
    'd', 'e', 'f', 'g', 'h', 'i', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'a', 'b', 'c', 'd', 'e', 'f',
    'g', 'h', 'i', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'a', 'b', 'c',
    'd', 'e', 'f', 'g', 'h', 'i', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
};

const char SQ_Rank[90] = {
    '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1',
    '1', '1', '1', '2', '2', '2', '2', '2', '2', '2', '2', '2', '3', '3', '3',
    '3', '3', '3', '3', '3', '3', '4', '4', '4', '4', '4', '4', '4', '4', '4',
    '5', '5', '5', '5', '5', '5', '5', '5', '5', '6', '6', '6', '6', '6', '6',
    '6', '6', '6', '7', '7', '7', '7', '7', '7', '7', '7', '7', '8', '8', '8',
    '8', '8', '8', '8', '8', '8', '9', '9', '9', '9', '9', '9', '9', '9', '9',
};

int decode_hash_value(const Bytes& slice, std::string* key, std::string* value) {
  if (slice.size() < 2 * sizeof(int16_t)) {
    return -1;
  }
  // Decode as unsigned to avoid negative shifts/indexes.
  uint16_t encoded = 0;
  std::memcpy(&encoded, slice.data(), sizeof(uint16_t));
  std::size_t src = encoded >> 8;
  std::size_t dst = encoded & 0x7F;
  if (src >= 90 || dst >= 90) {
    return -1;
  }
  if (encoded & 0x80) {
    key->resize(5);
    (*key)[0] = SQ_File[src];
    (*key)[1] = SQ_Rank[src];
    (*key)[2] = SQ_File[dst];
    if (SQ_Rank[src] == '7')
      (*key)[3] = '8';
    else if (SQ_Rank[src] == '2')
      (*key)[3] = '1';
    else
      return -1;

    switch (SQ_Rank[dst]) {
    case '0':
      (*key)[4] = 'q';
      break;
    case '1':
      (*key)[4] = 'r';
      break;
    case '2':
      (*key)[4] = 'b';
      break;
    case '3':
      (*key)[4] = 'n';
      break;
    default:
      return -1;
    }
  } else {
    key->resize(4);
    (*key)[0] = SQ_File[src];
    (*key)[1] = SQ_Rank[src];
    (*key)[2] = SQ_File[dst];
    (*key)[3] = SQ_Rank[dst];
  }
  int16_t val = 0;
  std::memcpy(&val, slice.data() + sizeof(int16_t), sizeof(int16_t));
  if (val == 0) {
    *value = "0";
  } else {
    bool neg = val < 0;
    uint16_t u = neg ? static_cast<uint16_t>(-val) : static_cast<uint16_t>(val);
    char tmp[6]; // int16 range -32768..32767 -> max 6 characters
    int len = 0;
    while (u > 0) {
      tmp[len++] = static_cast<char>('0' + (u % 10));
      u /= 10;
    }
    if (neg) tmp[len++] = '-';
    value->resize(len);
    for (int i = 0; i < len; ++i) {
      (*value)[i] = tmp[len - 1 - i];
    }
  }
  return 0;
}

int get_hash_values(const Bytes& slice, std::vector<StrPair>& values) {
  if (slice.empty() || slice.size() % (2 * sizeof(int16_t)) != 0) {
    return 0;
  }
  values.reserve(slice.size() / (2 * sizeof(int16_t)));
  for (size_t i = 0; i < slice.size(); i += 2 * sizeof(int16_t)) {
    std::string elem_field, elem_value;
    if (decode_hash_value(Bytes(slice.data() + i, 2 * sizeof(int16_t)),
                          &elem_field, &elem_value) == 0) {
      values.push_back(std::make_pair(elem_field, elem_value));
    }
  }
  return 0;
}

} // namespace cdb

#endif // WITH_CDB
