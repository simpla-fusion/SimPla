//
// Created by salmon on 16-7-6.
//

#ifndef SIMPLA_SPPARALLEL_H
#define SIMPLA_SPPARALLEL_H

#include "sp_lite_def.h"
#include </usr/local/cuda/include/cuda_runtime.h>


void spParallelInitialize(int argc, char **argv);

void spParallelFinalize();


#ifdef HAS_CUDA
#   include "spParallelCUDA.h"
#else
#   include "spParallelCPU.h"
#endif

void spMPIDataTypeCreate(int count, int *array_of_displacements, int type_tag, MPI_Datatype *new_type);

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
