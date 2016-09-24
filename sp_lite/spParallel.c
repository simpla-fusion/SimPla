//
// Created by salmon on 16-7-20.
//
#include <assert.h>
#include <mpi.h>
#include "spParallel.h"
#include "spMPI.h"


int spParallelInitialize(int argc, char **argv)
{

    spMPIInitialize(argc, argv);

    spParallelDeviceInitialize(argc, argv);

    return SP_SUCCESS;
}

int spParallelFinalize()
{
    spParallelDeviceFinalize();

    spMPIFinalize();

    return SP_SUCCESS;
}

int spParallelGlobalBarrier()
{
    spMPIBarrier();
    return SP_SUCCESS;
};


int spParallelThreadBlockDecompose(size_type num_of_threads_per_block, size_type grid_dim[3], size_type block_dim[3])
{
    block_dim[0] = num_of_threads_per_block;
    block_dim[1] = 1;
    block_dim[2] = 1;

    while (block_dim[0] > grid_dim[0])
    {
        block_dim[0] /= 2;
        block_dim[1] *= 2;
    }

    while (block_dim[1] > grid_dim[1])
    {
        block_dim[1] /= 2;
        block_dim[2] *= 2;
    }
    for (int i = 0; i < 3; ++i)
    {
        size_type L = grid_dim[i];
        grid_dim[i] = L / block_dim[i];
        if (grid_dim[i] * block_dim[i] < L) { ++(grid_dim[i]); }
    }

    return SP_SUCCESS;

}