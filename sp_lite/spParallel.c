//
// Created by salmon on 16-7-20.
//
#include <assert.h>
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

int spMPIDataTypeCreate(int type_tag, int type_size_in_byte, MPI_Datatype *new_type)
{
    *new_type = MPI_BYTE;
    switch (type_tag)
    {
        case SP_TYPE_float:
            *new_type = MPI_FLOAT;
            break;
        case SP_TYPE_double:
            *new_type = MPI_DOUBLE;
            break;

        case SP_TYPE_int:
            *new_type = MPI_INT;
            break;

        case SP_TYPE_long:
            *new_type = MPI_LONG;
            break;
        case SP_TYPE_int64_t:
            *new_type = MPI_INT64_T;
            break;
        default:
            *new_type = MPI_DATATYPE_NULL;
            break;
    }
    if (*new_type != MPI_DATATYPE_NULL)
    {
        MPI_Count t_size = 0;
        MPI_ERROR(MPI_Type_size_x(*new_type, &t_size));
        assert(t_size == type_size_in_byte);
    }


    return MPI_SUCCESS;
    //    MPI_ERROR(MPI_Type_create_indexed_block(count,
//                                            SP_NUMBER_OF_ENTITIES_IN_PAGE,
//                                            array_of_displacements,
//                                            ele_type,
//                                            new_type));
//    MPI_ERROR(MPI_Type_commit(new_type));
}

int spNdArrayUpdateHalo(void *buffer,
                        int ndims,
                        const size_type *shape,
                        const size_type *start,
                        const size_type *stride,
                        const size_type *count,
                        const size_type *block,
                        MPI_Datatype ele_type,
                        MPI_Comm comm)
{
    {
        int tope_type = MPI_CART;
        MPI_ERROR(MPI_Topo_test(comm, &tope_type));
        assert(tope_type == MPI_CART);
    }


    int mpi_topology_ndims = 0;

    MPI_Cartdim_get(comm, &mpi_topology_ndims);

    assert(mpi_topology_ndims <= ndims);

    int num_of_neighbour = 2 * mpi_topology_ndims;

    int mpi_sendrecv_count[num_of_neighbour];
    MPI_Datatype send_types[num_of_neighbour];
    MPI_Datatype recv_types[num_of_neighbour];
    MPI_Aint send_displs[num_of_neighbour];
    MPI_Aint recv_displs[num_of_neighbour];
    MPI_Count ele_size_in_byte = 0;

    MPI_ERROR(MPI_Type_size_x(ele_type, &ele_size_in_byte));

    int dims[ndims];

    for (int i = 0; i < ndims; ++i) { dims[i] = (int) shape[i]; }

    int s_count_lower[ndims];
    int s_start_lower[ndims];
    int s_count_upper[ndims];
    int s_start_upper[ndims];

    int r_count_lower[ndims];
    int r_start_lower[ndims];
    int r_count_upper[ndims];
    int r_start_upper[ndims];

    for (int d = 0; d < mpi_topology_ndims; ++d)
    {
        mpi_sendrecv_count[d] = mpi_sendrecv_count[2 * d + 1] = 1;

        for (int i = 0; i < ndims; ++i)
        {
            if (i < d)
            {
                s_count_lower[i] = (int) dims[i];
                s_start_lower[i] = 0;
                s_count_upper[i] = (int) dims[i];
                s_start_upper[i] = 0;

                r_count_lower[i] = (int) dims[i];
                r_start_lower[i] = 0;
                r_count_upper[i] = (int) dims[i];
                r_start_upper[i] = 0;
            }
            else if (i == d)
            {
                s_count_lower[i] = (int) start[i];
                s_start_lower[i] = (int) start[i];
                s_count_upper[i] = (int) (dims[i] - count[i] - start[i]);
                s_start_upper[i] = (int) (start[i] + count[i] - s_count_upper[i]);

                r_count_lower[i] = (int) start[i];
                r_start_lower[i] = (int) 0;
                r_count_upper[i] = (int) (dims[i] - count[i] - start[i]);
                r_start_upper[i] = (int) dims[i] - s_count_upper[i];
            }
            else
            {
                s_count_lower[i] = (int) count[i];
                s_start_lower[i] = (int) start[i];
                s_count_upper[i] = (int) count[i];
                s_start_upper[i] = (int) start[i];

                r_count_lower[i] = (int) count[i];
                r_start_lower[i] = (int) start[i];
                r_count_upper[i] = (int) count[i];
                r_start_upper[i] = (int) start[i];
            };
        }


        MPI_Type_create_subarray(ndims,
                                 dims,
                                 s_count_lower,
                                 s_start_lower,
                                 MPI_ORDER_C,
                                 ele_type,
                                 &send_types[2 * d + 0]);
        MPI_Type_create_subarray(ndims,
                                 dims,
                                 s_count_upper,
                                 s_start_upper,
                                 MPI_ORDER_C,
                                 ele_type,
                                 &send_types[2 * d + 1]);


        MPI_Type_create_subarray(ndims,
                                 dims,
                                 r_count_lower,
                                 r_start_lower,
                                 MPI_ORDER_C,
                                 ele_type,
                                 &recv_types[2 * d + 0]);
        MPI_Type_create_subarray(ndims,
                                 dims,
                                 r_count_upper,
                                 r_start_upper,
                                 MPI_ORDER_C,
                                 ele_type,
                                 &recv_types[2 * d + 1]);

        MPI_Type_commit(&(send_types[2 * d + 0]));
        MPI_Type_commit(&(send_types[2 * d + 1]));
        MPI_Type_commit(&(recv_types[2 * d + 0]));
        MPI_Type_commit(&(recv_types[2 * d + 1]));

        send_displs[2 * d + 0] = 0;
        send_displs[2 * d + 1] = 0;
        recv_displs[2 * d + 0] = 0;
        recv_displs[2 * d + 1] = 0;
    }


    spMPINeighborAllToAllCart(buffer, mpi_sendrecv_count, send_displs, send_types,
                              buffer, mpi_sendrecv_count, recv_displs, recv_types, comm);

    for (int i = 0; i < num_of_neighbour; ++i)
    {
        MPI_Type_free(&send_types[i]);
        MPI_Type_free(&recv_types[i]);
    }

    return MPI_SUCCESS;

}

/**
 * MPI_Neighbor_alltoallw
 *
 * @param send_buffer
 * @param send_counts
 * @param send_displs
 * @param send_types
 * @param recv_buffer
 * @param recv_counts
 * @param recv_displs
 * @param recv_types
 * @param comm
 * @return
 */
int spMPINeighborAllToAllCart(const void *send_buffer,
                              const int *send_counts,
                              const MPI_Aint *send_displs,
                              MPI_Datatype const *send_types,
                              void *recv_buffer,
                              const int *recv_counts,
                              const MPI_Aint *recv_displs,
                              MPI_Datatype const *recv_types,
                              MPI_Comm comm)
{
    int tag = 0;
    int mpi_topology_ndims = 0;
    MPI_ERROR(MPI_Cartdim_get(comm, &mpi_topology_ndims));

    for (int d = 0; d < mpi_topology_ndims; ++d)
    {
        int r0, r1;

        MPI_ERROR(MPI_Cart_shift(comm, d, 1, &r0, &r1));

        MPI_ERROR(MPI_Sendrecv(
            (byte_type *) (send_buffer) + send_displs[d * 2 + 0],
            send_counts[d],
            send_types[d * 2 + 0],
            r0,
            tag,
            (byte_type *) (recv_buffer) + recv_displs[d * 2 + 0],
            recv_counts[d],
            recv_types[d * 2 + 0],
            r1,
            tag,
            comm,
            MPI_STATUS_IGNORE));

        MPI_ERROR(MPI_Sendrecv(
            (byte_type *) (send_buffer) + send_displs[d * 2 + 1],
            send_counts[d],
            send_types[d * 2 + 1],
            r1,
            tag,
            (byte_type *) (recv_buffer) + recv_displs[d * 2 + 1],
            recv_counts[d],
            recv_types[d * 2 + 1],
            r0,
            tag,
            comm,
            MPI_STATUS_IGNORE));
    }

    return MPI_SUCCESS;

}
