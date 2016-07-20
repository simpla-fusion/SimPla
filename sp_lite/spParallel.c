//
// Created by salmon on 16-7-20.
//
#include "spParallel.h"

void spParallelInitialize(int argc, char **argv)
{

    spMPIInitialize(argc, argv);

    int num_of_device = 0;
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&num_of_device));
    CUDA_CHECK_RETURN(cudaSetDevice(spMPIProcessNum() % num_of_device));
    CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
    CUDA_CHECK_RETURN(cudaGetLastError());
}

void spParallelFinalize()
{
    CUDA_CHECK_RETURN(cudaDeviceReset());
    spMPIFinialize();
}
