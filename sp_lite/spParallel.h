//
// Created by salmon on 16-7-6.
//

#ifndef SIMPLA_SPPARALLEL_H
#define SIMPLA_SPPARALLEL_H

#include "sp_lite_def.h"

void spParallelInitialize();

void spParallelFinalize();

void spParallelGlobalSync();

void spParallelDeviceMalloc(void **, size_type s);

void spParallelDeviceFree(void *);

void spParallelMemcpy(void *dest, void const *src, size_type s);

void spParallelMemcpyToSymbol(void *dest, void const *src, size_type s);

void spParallelMemset(void *dest, byte_type v, size_type s);

#endif //SIMPLA_SPPARALLEL_H
