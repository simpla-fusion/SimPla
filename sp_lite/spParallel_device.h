//
// Created by salmon on 16-8-4.
//

#ifndef SIMPLA_SPPARALLELDEVICE_H
#define SIMPLA_SPPARALLELDEVICE_H

#ifdef __CUDACC__
#   include "cuda/spParallelCUDA.h"
#else

#   include "cpu/spParallelCPU.h"

#endif
#endif //SIMPLA_SPPARALLELDEVICE_H
