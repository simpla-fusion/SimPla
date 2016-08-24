//
// Created by salmon on 16-7-6.
//

#ifndef SIMPLA_SPPARALLEL_H
#define SIMPLA_SPPARALLEL_H

#include "sp_lite_def.h"
#include "../src/sp_capi.h"

int spParallelInitialize(int argc, char **argv);

int spParallelFinalize();

int spParallelDeviceInitialize(int argc, char **argv);

int spParallelDeviceFinalize();

int spParallelDeviceAlloc(void **, size_type);

int spParallelDeviceFree(void **);

int spParallelHostAlloc(void **, size_type);

int spParallelHostFree(void **);

int spParallelMemcpy(void *, void const *, size_type);

int spParallelMemcpyToCache(const void *, void const *, size_type);

int spParallelMemset(void *, int v, size_type);

int spParallelDeviceSync();

int spParallelGlobalBarrier();

int spParallelAssign(size_type num_of_point, size_type *offset, Real *d, Real const *v);

int spParallelUpdateNdArrayHalo(int num_of_buffer, void **buffers, const spDataType *ele_type, int ndims,
                                const size_type *dims, const size_type *start, const size_type *,
                                const size_type *count, const size_type *, int mpi_sync_start_dims);

int spParallelUpdateNdArrayHalo2(int num_of_buffer,
                                 void **buffers,
                                 const spDataType *data_desc,
                                 int ndims,
                                 const size_type *shape,
                                 const size_type *start,
                                 const size_type *stride,
                                 const size_type *count,
                                 const size_type *block,
                                 int mpi_sync_start_dims);

int spParallelDeviceFillInt(int *d, int v, size_type s);

int spParallelDeviceFillReal(Real *d, Real v, size_type s);

int spParallelScan(size_type *, size_type num);

int spParallelGridDim();

int spParallelBlockDim();

int spParallelThreadBlockDecompose(size_type num_of_threads_per_block,
                                   unsigned int ndims,
                                   size_type const *min,
                                   size_type const *max,
                                   size_type grid_dim[3],
                                   size_type block_dim[3]);

#endif //SIMPLA_SPPARALLEL_H
