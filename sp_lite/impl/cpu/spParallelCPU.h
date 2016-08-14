//
// Created by salmon on 16-7-20.
//

#ifndef SIMPLA_SPPARALLELCPU_H
#define SIMPLA_SPPARALLELCPU_H

#include <stdlib.h>
#include <string.h>

#include "../../sp_lite_def.h"

typedef struct { Real x, y, z; } Real3;
typedef struct { size_type x, y, z; } dim3;

#define INLINE  static inline
#define DEVICE
#define HOST
#define SP_DEVICE_DECLARE_KERNEL(_FUN_, ...)   void _FUN_ (dim3 threadIdx,dim3 blockDim,dim3 blockIdx,dim3 gridDim, __VA_ARGS__)

#define SP_DEVICE_CALL_KERNEL(_FUN_, _DIMS_, _N_THREADS_, ...)       \
   { dim3 threadIdx={0,0,0};dim3 blockIdx={1,1,1};                                          \
    _FUN_(threadIdx , _N_THREADS_,blockIdx, _DIMS_ , __VA_ARGS__) ; } \

#endif //SIMPLA_SPPARALLELCPU_H
