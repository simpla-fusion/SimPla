//
// Created by salmon on 16-7-20.
//

#ifndef SIMPLA_SPPARALLELCPU_H
#define SIMPLA_SPPARALLELCPU_H

#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK_RETURN(...)

#define spParallelDeviceAlloc(_P_, _S_)      {*_P_ = malloc(_S_);}

#define spParallelDeviceFree(_P_)      if (*_P_ != NULL) { (free(*_P_)); *_P_ = NULL;   };

#define spParallelHostAlloc(_P_, _S_)    (*_P_=malloc(_S_))

#define spParallelHostFree(_P_)  if (*_P_ != NULL) { free(*_P_);  *_P_ = NULL; }

#define spParallelMemcpy(_D_, _S_, _N_) (memcpy(_D_, _S_,(_N_)));

#define  spParallelMemcpyToSymbol(_D_, _S_, _N_)    (memcpy(_D_, _S_, _N_));

#define spParallelMemset(_D_, _V_, _N_)  (memset(_D_, _V_, _N_));

#define spParallelDeviceSync()  {MPI_Barrier(MPI_COMM_WORLD);}

typedef struct Real3_s
{
    Real x, y, z;
} Real3;
#endif //SIMPLA_SPPARALLELCPU_H
