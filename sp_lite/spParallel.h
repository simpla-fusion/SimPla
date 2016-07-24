//
// Created by salmon on 16-7-6.
//

#ifndef SIMPLA_SPPARALLEL_H
#define SIMPLA_SPPARALLEL_H

#include "sp_lite_def.h"
#include "../src/sp_capi.h"

int spParallelInitialize(int argc, char **argv);

int spParallelFinalize();


#ifndef __CUDACC__
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
                                const size_type *);


#endif //SIMPLA_SPPARALLEL_H
