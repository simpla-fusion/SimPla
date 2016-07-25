//
// Created by salmon on 16-7-20.
//
#include <assert.h>
#include "spParallel.h"

dim3 sizeType2Dim3(size_type const *v)
{
    dim3 res;
    res.x = (int) v[0];
    res.y = (int) v[1];
    res.z = (int) v[2];
    return res;
}

int spParallelInitialize(int argc, char **argv)
{

    spMPIInitialize(argc, argv);

    int num_of_device = 0;
    SP_PARALLEL_CHECK_RETURN(cudaGetDeviceCount(&num_of_device));
    SP_PARALLEL_CHECK_RETURN(cudaSetDevice(spMPIRank() % num_of_device));
    SP_PARALLEL_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete


    SP_PARALLEL_CHECK_RETURN(cudaGetLastError());
    return SP_SUCCESS;

}

int spParallelFinalize()
{

    SP_PARALLEL_CHECK_RETURN(cudaDeviceReset());
    spMPIFinialize();
    return SP_SUCCESS;

}
int spParallelGlobalBarrier()
{
    spMPIBarrier();
    return SP_SUCCESS;
};


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
int spMPINeighborAllToAll(const void *send_buffer,
                          const int *send_counts,
                          const MPI_Aint *send_displs,
                          MPI_Datatype const *send_types,
                          void *recv_buffer,
                          const int *recv_counts,
                          const MPI_Aint *recv_displs,
                          MPI_Datatype const *recv_types,
                          MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) { return SP_FAILED; }
    else
    {
        int tope_type = MPI_CART;

        MPI_ERROR(MPI_Topo_test(comm, &tope_type));

        if (tope_type != MPI_CART) { return SP_FAILED; }
    }

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

    return SP_SUCCESS;

}
//
//int testNdArrayUpdateHalo()
//{
//    MPI_Comm comm = spMPIComm();
//
//    int rank = spMPIRank();
//
//    int size = spMPISize();
//
//    int buffer[25] = {
//        rank, rank, rank, rank, rank,
//        rank, rank, rank, rank, rank,
//        rank, rank, rank, rank, rank,
//        rank, rank, rank, rank, rank,
//        rank, rank, rank, rank, rank
//    };
//
//    size_type local_dims[2] = {5, 5};
//    size_type start[2] = {1, 1};
//    size_type count[2] = {3, 3};
//
//    spParallelUpdateNdArrayHalo(buffer, 2, local_dims, start, NULL, count, NULL, MPI_INT);
//
//
//    printf("\n"
//               "[%d/%d/%d] \t  %d,%d,%d,%d,%d \n"
//               "           \t  %d,%d,%d,%d,%d \n"
//               "           \t  %d,%d,%d,%d,%d \n"
//               "           \t  %d,%d,%d,%d,%d \n"
//               "           \t  %d,%d,%d,%d,%d \n", rank, size, 6,
//           buffer[0], buffer[1], buffer[2], buffer[3], buffer[4],
//           buffer[5], buffer[6], buffer[7], buffer[8], buffer[9],
//           buffer[10], buffer[11], buffer[12], buffer[13], buffer[14],
//           buffer[15], buffer[16], buffer[17], buffer[18], buffer[19],
//           buffer[20], buffer[21], buffer[22], buffer[23], buffer[24]
//    );
//}

int spParallelUpdateNdArrayHalo(void *buffer,
                                const struct spDataType_s *data_desc,
                                int ndims,
                                const size_type *shape,
                                const size_type *start,
                                const size_type *stride,
                                const size_type *count,
                                const size_type *block)
{
    MPI_Comm comm = spMPIComm();

    if (comm == MPI_COMM_NULL) { return SP_SUCCESS; }
    else
    {
        int tope_type = MPI_CART;
        MPI_ERROR(MPI_Topo_test(comm, &tope_type));
        assert(tope_type == MPI_CART);
    }

    MPI_Datatype const ele_type = *spDataTypeMPIType(data_desc);

    if (ele_type == MPI_DATATYPE_NULL) { return SP_FAILED; }

    int mpi_topology_ndims = 0;

    MPI_ERROR(MPI_Cartdim_get(comm, &mpi_topology_ndims));

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
        mpi_sendrecv_count[2 * d] = mpi_sendrecv_count[2 * d + 1] = 1;

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


        MPI_ERROR(MPI_Type_create_subarray(ndims,
                                           dims,
                                           s_count_lower,
                                           s_start_lower,
                                           MPI_ORDER_C,
                                           ele_type,
                                           &send_types[2 * d + 0]));
        MPI_ERROR(MPI_Type_create_subarray(ndims,
                                           dims,
                                           s_count_upper,
                                           s_start_upper,
                                           MPI_ORDER_C,
                                           ele_type,
                                           &send_types[2 * d + 1]));


        MPI_ERROR(MPI_Type_create_subarray(ndims,
                                           dims,
                                           r_count_lower,
                                           r_start_lower,
                                           MPI_ORDER_C,
                                           ele_type,
                                           &recv_types[2 * d + 0]));
        MPI_ERROR(MPI_Type_create_subarray(ndims,
                                           dims,
                                           r_count_upper,
                                           r_start_upper,
                                           MPI_ORDER_C,
                                           ele_type,
                                           &recv_types[2 * d + 1]));

        MPI_ERROR(MPI_Type_commit(&(send_types[2 * d + 0])));
        MPI_ERROR(MPI_Type_commit(&(send_types[2 * d + 1])));
        MPI_ERROR(MPI_Type_commit(&(recv_types[2 * d + 0])));
        MPI_ERROR(MPI_Type_commit(&(recv_types[2 * d + 1])));

        send_displs[2 * d + 0] = 0;
        send_displs[2 * d + 1] = 0;
        recv_displs[2 * d + 0] = 0;
        recv_displs[2 * d + 1] = 0;
    }


    SP_CHECK_RETURN(spMPINeighborAllToAll(buffer, mpi_sendrecv_count, send_displs, send_types,
                                          buffer, mpi_sendrecv_count, recv_displs, recv_types, comm));

    for (int i = 0; i < num_of_neighbour; ++i)
    {
        MPI_ERROR(MPI_Type_free(&send_types[i]));
        MPI_ERROR(MPI_Type_free(&recv_types[i]));
    }

    return MPI_SUCCESS;

}

int spUpdateIndexedBlock(void const *send_buffer,
                         const size_type **send_disp_s,
                         const size_type *send_block_count,
                         void *recv_buffer,
                         const size_type **recv_disp_s,
                         const size_type *recv_block_count,
                         size_type block_length,
                         MPI_Datatype ele_type,
                         MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) { return SP_FAILED; }

    int tag = 0;

    int mpi_topology_ndims = 0;

    MPI_ERROR(MPI_Cartdim_get(comm, &mpi_topology_ndims));

    int num_of_neighbour = mpi_topology_ndims * 2;

    int mpi_sendrecv_count[num_of_neighbour];

    MPI_Aint send_displs[num_of_neighbour], recv_displs[num_of_neighbour];

    MPI_Datatype send_types[num_of_neighbour];
    MPI_Datatype recv_types[num_of_neighbour];

    for (int i = 0; i < mpi_topology_ndims; ++i)
    {
        int *disp = NULL;

#define CPY(_COUNT_, _P_)                                                         \
        free(disp);disp = malloc(send_block_count[2 * i + 0] * sizeof(int));      \
        for (size_type s = 0; s < _COUNT_; ++s) { disp[s] = (int) (_P_[s]); }

        CPY(send_block_count[2 * i + 0], send_disp_s[2 * i + 0]);
        MPI_Type_create_indexed_block((int) send_block_count[2 * i + 0], (int) block_length,
                                      disp, ele_type, &send_types[2 * i + 0]);

        CPY(send_block_count[2 * i + 1], send_disp_s[2 * i + 1]);
        MPI_Type_create_indexed_block((int) send_block_count[2 * i + 1], (int) block_length,
                                      disp, ele_type, &send_types[2 * i + 1]);


        CPY(recv_block_count[2 * i + 0], recv_disp_s[2 * i + 0]);
        MPI_Type_create_indexed_block((int) recv_block_count[2 * i + 0], (int) block_length,
                                      disp, ele_type, &recv_types[2 * i + 0]);


        CPY(recv_block_count[2 * i + 1], recv_disp_s[2 * i + 1]);
        MPI_Type_create_indexed_block((int) recv_block_count[2 * i + 1], (int) block_length,
                                      disp, ele_type, &recv_types[2 * i + 1]);

        free(disp);
#undef CPY

        mpi_sendrecv_count[2 * i + 0] = mpi_sendrecv_count[2 * i + 1] = 1;
        send_displs[2 * i + 0] = recv_displs[2 * i + 0] = 0;
        send_displs[2 * i + 1] = recv_displs[2 * i + 1] = 0;
    }

    spMPINeighborAllToAll(send_buffer, mpi_sendrecv_count, send_displs, send_types,
                          recv_buffer, mpi_sendrecv_count, recv_displs, recv_types, comm);

    for (int i = 0; i < num_of_neighbour; ++i)
    {
        MPI_Type_free(&send_types[i]);
        MPI_Type_free(&recv_types[i]);
    }
    return MPI_SUCCESS;
}