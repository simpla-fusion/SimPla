//
// Created by salmon on 16-7-6.
//

#ifndef SIMPLA_SPPARALLEL_H
#define SIMPLA_SPPARALLEL_H

#include "sp_lite_def.h"

int spParallelInitialize(int argc, char **argv);

int spParallelFinalize();

int spParallelDeviceInitialize(int argc, char **argv);

int spParallelDeviceFinalize();

int spParallelDeviceAlloc(void **, int);

int spParallelDeviceFree(void **);

int spParallelHostAlloc(void **, int);

int spParallelHostFree(void **);

int spParallelMemcpy(void *, void const *, int);

int spParallelMemcpyToCache(const void *, void const *, int);

int spParallelMemset(void *, int v, int);

int spParallelDeviceSync();

int spParallelGlobalBarrier();

int spParallelAssign(int num_of_point, int *offset, Real *d, Real const *v);

int spParallelDeviceFillInt(int *d, int v, int s);

int spParallelDeviceFillReal(Real *d, Real v, int s);

int spParallelGridDim();

int spParallelBlockDim();

int spParallelThreadBlockDecompose(int num_of_threads_per_block,
                                   unsigned int ndims,
                                   const int *min,
                                   const int *max,
                                   int *grid_dim,
                                   int *block_dim);

int spMemoryDeviceToHost(void **p, void *src, int size_in_byte);

int spMemoryHostFree(void **p);

#endif //SIMPLA_SPPARALLEL_H
