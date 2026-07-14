#pragma once

#include <cstddef>

#include "nnue_macros.h"
#include "training_data_loader_structs.h"

// Opaque handles
struct SparseBatch;
struct FenBatch;
struct FenBatchStream;
struct SparseBatchStream;

// batches
NNUE_API SparseBatch* NNUE_CDECL get_sparse_batch_from_fens(const char*        feature_set_c,
                                        int                num_fens,
                                        const char* const* fens,
                                        int*               scores,
                                        int*               plies,
                                        int*               results);

NNUE_API void NNUE_CDECL destroy_sparse_batch(SparseBatch* e);
NNUE_API void NNUE_CDECL destroy_fen_batch(FenBatch* e);

// fen stream
NNUE_API FenBatchStream* NNUE_CDECL create_fen_batch_stream(int                  concurrency,
                                        int                  num_files,
                                        const char* const*   filenames,
                                        int                  batch_size,
                                        bool                 cyclic,
                                        DataloaderSkipConfig config,
                                        DataloaderDDPConfig  ddp_config);

NNUE_API void      NNUE_CDECL destroy_fen_batch_stream(FenBatchStream* stream);
NNUE_API FenBatch* NNUE_CDECL fetch_next_fen_batch(FenBatchStream* stream);

// sparse stream
NNUE_API SparseBatchStream* NNUE_CDECL create_sparse_batch_stream(const char*          feature_set_c,
                                              int                  concurrency,
                                              int                  num_files,
                                              const char* const*   filenames,
                                              int                  batch_size,
                                              bool                 cyclic,
                                              DataloaderSkipConfig config,
                                              DataloaderDDPConfig  ddp_config);

NNUE_API SparseBatchStream* NNUE_CDECL create_cdb_sparse_batch_stream(const char*          feature_set_c,
                                                  const char*          db_path,
                                                  int                  concurrency,
                                                  int                  batch_size,
                                                  bool                 cyclic,
                                                  DataloaderSkipConfig config,
                                                  DataloaderDDPConfig  ddp_config);

NNUE_API void         NNUE_CDECL destroy_sparse_batch_stream(SparseBatchStream* stream);
NNUE_API SparseBatch* NNUE_CDECL fetch_next_sparse_batch(SparseBatchStream* stream);

// Explicit global shutdown.  Destroys all streams that have not already been
// destroyed.  This is safe to call from a Python atexit handler before the
// interpreter starts tearing down modules; each stream is deleted at most once
// even if Python wrappers also call destroy from __del__.
NNUE_API void NNUE_CDECL nnue_data_loader_shutdown();
