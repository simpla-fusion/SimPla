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


int spParallelThreadBlockDecompose(size_type num_of_threads_per_block,
                                   unsigned int ndims,
                                   size_type const *min,
                                   size_type const *max,
                                   size_type grid_dim[3],
                                   size_type block_dim[3])
{
    assert(max[0] > min[0]);
    assert(max[1] > min[1]);
    assert(max[2] > min[2]);


    block_dim[0] = num_of_threads_per_block;
    block_dim[1] = 1;
    block_dim[2] = 1;

    while (block_dim[0] + min[0] > max[0])
    {
        block_dim[0] /= 2;
        block_dim[1] *= 2;
    }

    while (block_dim[1] + min[1] > max[1])
    {
        block_dim[1] /= 2;
        block_dim[2] *= 2;
    }
    grid_dim[0] = (max[0] - min[0]) / block_dim[0];
    grid_dim[1] = (max[1] - min[1]) / block_dim[1];
    grid_dim[2] = (max[2] - min[2]) / block_dim[2];

    grid_dim[0] = (grid_dim[0] * block_dim[0] < max[0] - min[0]) ? grid_dim[0] + 1 : grid_dim[0];
    grid_dim[1] = (grid_dim[1] * block_dim[1] < max[1] - min[1]) ? grid_dim[1] + 1 : grid_dim[1];
    grid_dim[2] = (grid_dim[2] * block_dim[2] < max[2] - min[2]) ? grid_dim[2] + 1 : grid_dim[2];

    assert(grid_dim[0] * block_dim[0] >= max[0] - min[0]);
    assert(grid_dim[1] * block_dim[1] >= max[1] - min[1]);
    assert(grid_dim[2] * block_dim[2] >= max[2] - min[2]);
    return SP_SUCCESS;

}