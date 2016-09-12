//
// Created by salmon on 16-7-20.
//
#include <assert.h>
#include "spParallel.h"


int spParallelInitialize(int argc, char **argv)
{

//    MPI_CALL(MPI_Init(&argc, &argv));
//
//    int m_num_process_;
//
//    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &m_num_process_));
//    int mpi_rank, mpi_size;
//    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
//    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
//
//    int mpi_topology_ndims = 3;
//
//    int m_topology_dims_[3] = {1, 1, 1};
//
//    int m_topology_coord_[3] = {0, 0, 0};
//
//    int periods[3] = {0, 0, 0};
//
//    MPI_CALL(MPI_Dims_create(m_num_process_, mpi_topology_ndims, m_topology_dims_));
//
//
//    for (int i = 0; i < mpi_topology_ndims; ++i) { periods[i] = SP_TRUE; }
//
//    MPI_CALL(MPI_Cart_create(MPI_COMM_WORLD, mpi_topology_ndims,
//                             m_topology_dims_, periods, MPI_ORDER_C, &spMPIComm()));
//
//
//    MPI_CALL(MPI_Cart_coords(spMPIComm(), mpi_rank, mpi_topology_ndims, m_topology_coord_));
    spMPIInitialize(argc, argv);
    spParallelDeviceInitialize(argc, argv);
    return SP_SUCCESS;
}

int spParallelFinalize()
{
    spParallelDeviceFinalize();
    spMPIFinalize();
//    if (spMPIComm() != MPI_COMM_NULL)
//    {
//
//        MPI_CALL(MPI_Finalize());
//
//        spMPIComm() = MPI_COMM_NULL;
//
//    }
    return SP_SUCCESS;
}

int spParallelGlobalBarrier()
{
    spMPIBarrier();
    return SP_SUCCESS;
};


int spParallelThreadBlockDecompose(int num_of_threads_per_block,
                                   unsigned int ndims,
                                   const int *min,
                                   const int *max,
                                   int *grid_dim,
                                   int *block_dim)
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

}