//
// Created by salmon on 16-7-6.
//

#ifndef SIMPLA_SPPARALLEL_H
#define SIMPLA_SPPARALLEL_H

#include "sp_lite_def.h"
#include </usr/local/cuda/include/cuda_runtime.h>


void spParallelInitialize(int argc, char **argv);

void spParallelFinalize();

//void spParallelDeviceSync();
//
//void spParallelHostMalloc(void **, size_type s);
//
//void spParallelHostFree(void **);


#ifndef DISABLE_CUDA
#define CUDA_CHECK_RETURN(_CMD_) {                                            \
    cudaError_t _m_cudaStat = _CMD_;                                        \
    if (_m_cudaStat != cudaSuccess) {                                        \
         printf("Error [code=0x%x] %s at line %d in file %s\n",                    \
                _m_cudaStat,cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);        \
        exit(1);                                                            \
    } }
#define spParallelDeviceMalloc(_P_, _S_)      CUDA_CHECK_RETURN(cudaMalloc(_P_, _S_));

#define spParallelDeviceFree(_P_)      if (*_P_ != NULL) { CUDA_CHECK_RETURN(cudaFree(*_P_)); *_P_ = NULL;   };

#define spParallelMemcpy(_D_, _S_, _N_) CUDA_CHECK_RETURN(cudaMemcpy(_D_, _S_,(_N_), cudaMemcpyDefault));

#define  spParallelMemcpyToSymbol(_D_, _S_, _N_)    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(_D_, _S_, _N_));

#define spParallelMemset(_D_, _V_, _N_)  CUDA_CHECK_RETURN(cudaMemset(_D_, _V_, _N_));

#define spParallelDeviceSync()    CUDA_CHECK_RETURN(cudaDeviceSynchronize())

#define spParallelHostMalloc(_P_, _S_)    CUDA_CHECK_RETURN(cudaHostAlloc(_P_, _S_, cudaHostAllocDefault))

#define spParallelHostFree(_P_)  if (*_P_ != NULL) { cudaFreeHost(*_P_);  *_P_ = NULL; }
#endif // DISABLE_CUDA


#define MPI_ERROR(_CMD_)                                                   \
{                                                                          \
    int _mpi_error_code_ = _CMD_;                                          \
    if (_mpi_error_code_ != MPI_SUCCESS)                                   \
    {                                                                      \
        char _error_msg[MPI_MAX_ERROR_STRING];                             \
        MPI_Error_string(_mpi_error_code_, _error_msg, NULL);           \
        ERROR(_error_msg);                                                 \
    }                                                                      \
}


#endif //SIMPLA_SPPARALLEL_H
