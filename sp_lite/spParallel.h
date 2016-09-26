//
// Created by salmon on 16-7-6.
//

#ifndef SIMPLA_SPPARALLEL_H
#define SIMPLA_SPPARALLEL_H

#include "sp_config.h"

int spParallelInitialize(int argc, char **argv);

int spParallelFinalize();

int spParallelDeviceInitialize(int argc, char **argv);

int spParallelDeviceFinalize();

int spMemoryDeviceAlloc(void **, size_type);

int spMemoryDeviceFree(void **);

int spMemHostAlloc(void **, size_type);

int spMemHostFree(void **);

int spMemoryCopy(void *, void const *, size_type);

int spMemCopyToCache(const void *, void const *, size_type);

int spMemSet(void *, int v, size_type);

int spMemoryDeviceToHost(void **p, void *src, size_type size_in_byte);

int spMemoryHostFree(void **p);

int spParallelDeviceSync();

int spParallelGlobalBarrier();

int spParallelAssign(size_type num_of_point, size_type *offset, Real *d, Real const *v);

int spParallelDeviceFillInt(int *d, int v, size_type s);

int spParallelDeviceFillReal(Real *d, Real v, size_type s);
//
//int spParallelGridDim();
//
//int spParallelBlockDim();

int spParallelThreadBlockDecompose(size_type num_of_threads_per_block, size_type grid_dim[3], size_type block_dim[3]);





#endif //SIMPLA_SPPARALLEL_H
