//
// Created by salmon on 16-7-6.
//

#ifndef SIMPLA_SPPARALLEL_H
#define SIMPLA_SPPARALLEL_H

#include "sp_lite_def.h"
#include "../src/sp_capi.h"

int spParallelInitialize(int argc, char **argv);

int spParallelFinalize();


#ifndef USE_CUDA
#   include "spParallelCPU.h"
#else
#   include "spParallelCUDA.h"
#endif
struct spDataType_s;

int spParallelUpdateNdArrayHalo(void *buffer,
                                const struct spDataType_s *ele_type,
                                int ndims,
                                const size_type *dims,
                                const size_type *start,
                                const size_type *,
                                const size_type *count,
                                const size_type *,
                                int mpi_sync_start_dims);

int spParallelGlobalBarrier();

int spParallelDeviceFillInt(int *d, int v, size_type s);

int spParallelDeviceFillReal(Real *d, Real v, size_type s);

#endif //SIMPLA_SPPARALLEL_H
