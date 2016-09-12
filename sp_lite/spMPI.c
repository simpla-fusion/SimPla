//
// Created by salmon on 16-9-12.
//
#include <assert.h>
#include "spMPI.h"
#include "sp_lite_def.h"

#define MPI_CALL(_CMD_)                                                   \
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
    if (comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }

    int tope_type = MPI_CART;

    MPI_CALL(MPI_Topo_test(comm, &tope_type));

    if (tope_type != MPI_CART) { return SP_FAILED; }


    int tag = 0;

    int mpi_topology_ndims = 0;

    MPI_CALL(MPI_Cartdim_get(comm, &mpi_topology_ndims));

    for (int d = 0; d < mpi_topology_ndims; ++d)
    {
        if (send_types[d * 2 + 0] == MPI_DATATYPE_NULL) { continue; }
        int r0, r1;

        MPI_CALL(MPI_Cart_shift(comm, d, 1, &r0, &r1));

        MPI_CALL(MPI_Sendrecv(
            (byte_type *) (send_buffer) + send_displs[d * 2 + 0],
            send_counts[d],
            send_types[d * 2 + 0],
            r0,
            tag,
            (byte_type *) (recv_buffer) + recv_displs[d * 2 + 0],
            recv_counts[d],
            recv_types[d * 2 + 0],
            r0,
            tag,
            comm,
            MPI_STATUS_IGNORE));

        MPI_CALL(MPI_Sendrecv(
            (byte_type *) (send_buffer) + send_displs[d * 2 + 1],
            send_counts[d],
            send_types[d * 2 + 1],
            r1,
            tag,
            (byte_type *) (recv_buffer) + recv_displs[d * 2 + 1],
            recv_counts[d],
            recv_types[d * 2 + 1],
            r1,
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
//    size_type m_dims_[2] = {5, 5};
//    size_type start[2] = {1, 1};
//    size_type count[2] = {3, 3};
//
//    spMPICartUpdateNdArrayHalo(buffer, 2, m_dims_, start, NULL, count, NULL, MPI_INT);
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

int spMPICartUpdateNdArrayHalo2(int num_of_buffer, void **buffers, const spDataType *data_desc, int ndims,
                                const size_type *shape, const size_type *start, const size_type *stride,
                                const size_type *count, const size_type *block, int mpi_sync_start_dims)
{
    MPI_Comm comm = spMPIComm();

    if (comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }

    int tope_type = MPI_CART;

    MPI_CALL(MPI_Topo_test(comm, &tope_type));

    assert(tope_type == MPI_CART);


    MPI_Datatype ele_type = spDataTypeMPIType(data_desc);

    if (ele_type == MPI_DATATYPE_NULL) { return SP_FAILED; }

    int mpi_topology_ndims = 0;

    MPI_CALL(MPI_Cartdim_get(comm, &mpi_topology_ndims));

    assert(mpi_topology_ndims <= ndims);

    int num_of_neighbour = 2 * mpi_topology_ndims;

    int mpi_sendrecv_count[num_of_neighbour];
    MPI_Datatype send_types[num_of_neighbour];
    MPI_Datatype recv_types[num_of_neighbour];
    MPI_Aint send_displs[num_of_neighbour];
    MPI_Aint recv_displs[num_of_neighbour];
    MPI_Count ele_size_in_byte = 0;

    MPI_CALL(MPI_Type_size_x(ele_type, &ele_size_in_byte));

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

        send_types[2 * d + 0] = MPI_DATATYPE_NULL;
        send_types[2 * d + 1] = MPI_DATATYPE_NULL;
        recv_types[2 * d + 0] = MPI_DATATYPE_NULL;
        recv_types[2 * d + 1] = MPI_DATATYPE_NULL;

        if (dims[d] == 1) { continue; }

        mpi_sendrecv_count[2 * d] = mpi_sendrecv_count[2 * d + 1] = 1;

        for (int i = 0; i < ndims; ++i)
        {
            if (i >= mpi_sync_start_dims && i < d + mpi_sync_start_dims)
            {
                s_count_lower[i] = dims[i];
                s_start_lower[i] = 0;
                s_count_upper[i] = dims[i];
                s_start_upper[i] = 0;

                r_count_lower[i] = dims[i];
                r_start_lower[i] = 0;
                r_count_upper[i] = dims[i];
                r_start_upper[i] = 0;
            }
            else if (i == d + mpi_sync_start_dims)
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


        MPI_CALL(MPI_Type_create_subarray(ndims,
                                          dims,
                                          s_count_upper,
                                          s_start_upper,
                                          MPI_ORDER_C,
                                          ele_type,
                                          &send_types[2 * d + 0]));

        MPI_CALL(MPI_Type_create_subarray(ndims,
                                          dims,
                                          s_count_lower,
                                          s_start_lower,
                                          MPI_ORDER_C,
                                          ele_type,
                                          &send_types[2 * d + 1]));

        MPI_CALL(MPI_Type_create_subarray(ndims,
                                          dims,
                                          r_count_lower,
                                          r_start_lower,
                                          MPI_ORDER_C,
                                          ele_type,
                                          &recv_types[2 * d + 0]));
        MPI_CALL(MPI_Type_create_subarray(ndims,
                                          dims,
                                          r_count_upper,
                                          r_start_upper,
                                          MPI_ORDER_C,
                                          ele_type,
                                          &recv_types[2 * d + 1]));

        MPI_CALL(MPI_Type_commit(&(send_types[2 * d + 0])));
        MPI_CALL(MPI_Type_commit(&(send_types[2 * d + 1])));
        MPI_CALL(MPI_Type_commit(&(recv_types[2 * d + 0])));
        MPI_CALL(MPI_Type_commit(&(recv_types[2 * d + 1])));

        send_displs[2 * d + 0] = 0;
        send_displs[2 * d + 1] = 0;
        recv_displs[2 * d + 0] = 0;
        recv_displs[2 * d + 1] = 0;
    }

    for (int i = 0; i < num_of_buffer; ++i)
    {
        SP_CALL(spMPINeighborAllToAll(buffers[i], mpi_sendrecv_count, send_displs, send_types,
                                      buffers[i], mpi_sendrecv_count, recv_displs, recv_types, comm));
    }


    for (int i = 0; i < num_of_neighbour; ++i)
    {
        if (send_types[i] != MPI_DATATYPE_NULL) MPI_CALL(MPI_Type_free(&send_types[i]));

        if (recv_types[i] != MPI_DATATYPE_NULL) MPI_CALL(MPI_Type_free(&recv_types[i]));
    }

    return SP_SUCCESS;

}

typedef struct spMPICartUpdater_s
{
    MPI_Comm comm;
    int num_of_neighbour;
    int mpi_sendrecv_count[6];
    MPI_Datatype send_types[6];
    MPI_Datatype recv_types[6];
    MPI_Aint send_displs[6];
    MPI_Aint recv_displs[6];
} spMPICartUpdater;

int spMPICartUpdaterDestroy(spMPICartUpdater **updater)
{
    for (int i = 0; i < (*updater)->num_of_neighbour; ++i)
    {
        if ((*updater)->send_types[i] != MPI_DATATYPE_NULL) MPI_CALL(MPI_Type_free(&((*updater)->send_types[i])));

        if ((*updater)->recv_types[i] != MPI_DATATYPE_NULL) MPI_CALL(MPI_Type_free(&((*updater)->recv_types[i])));
    }

    free(*updater);

}

int spMPICartUpdate(spMPICartUpdater const *updater, void *buffer)
{


    SP_CALL(spMPINeighborAllToAll(buffer,
                                  updater->mpi_sendrecv_count,
                                  updater->send_displs,
                                  updater->send_types,
                                  buffer,
                                  updater->mpi_sendrecv_count,
                                  updater->recv_displs,
                                  updater->recv_types,
                                  updater->comm));

    return SP_SUCCESS;

}

int spMPICartUpdateAll(spMPICartUpdater const *updater, int num_of_buffer, void **buffers)
{
    for (int i = 0; i < num_of_buffer; ++i) {SP_CALL(spMPICartUpdate(updater, buffers[i])); }
    return SP_SUCCESS;
}

int spMPICartUpdaterCreateDA(spMPICartUpdater **updater, const spDataType *data_desc, int ndims,
                             const size_type *shape, const size_type *start, const size_type *stride,
                             const size_type *count, const size_type *block, int mpi_sync_start_dims)
{
    *updater = malloc(sizeof(spMPICartUpdater));

    (*updater)->comm = spMPIComm();

    if ((*updater)->comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }

    int topo_type = MPI_CART;

    MPI_CALL(MPI_Topo_test((*updater)->comm, &topo_type));

    assert(topo_type == MPI_CART);


    MPI_Datatype ele_type = spDataTypeMPIType(data_desc);

    if (ele_type == MPI_DATATYPE_NULL) { return SP_FAILED; }

    int mpi_topology_ndims = 0;

    MPI_CALL(MPI_Cartdim_get((*updater)->comm, &mpi_topology_ndims));

    assert(mpi_topology_ndims <= ndims);


    MPI_Count ele_size_in_byte = 0;

    MPI_CALL(MPI_Type_size_x(ele_type, &ele_size_in_byte));

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

    (*updater)->num_of_neighbour = 6;

    for (int d = 0; d < mpi_topology_ndims; ++d)
    {

        (*updater)->send_types[2 * d + 0] = MPI_DATATYPE_NULL;
        (*updater)->send_types[2 * d + 1] = MPI_DATATYPE_NULL;
        (*updater)->recv_types[2 * d + 0] = MPI_DATATYPE_NULL;
        (*updater)->recv_types[2 * d + 1] = MPI_DATATYPE_NULL;

        if (dims[d] == 1) { continue; }

        (*updater)->mpi_sendrecv_count[2 * d] = (*updater)->mpi_sendrecv_count[2 * d + 1] = 1;

        for (int i = 0; i < ndims; ++i)
        {
            if (i >= mpi_sync_start_dims && i < d + mpi_sync_start_dims)
            {
                s_count_lower[i] = dims[i];
                s_start_lower[i] = 0;
                s_count_upper[i] = dims[i];
                s_start_upper[i] = 0;

                r_count_lower[i] = dims[i];
                r_start_lower[i] = 0;
                r_count_upper[i] = dims[i];
                r_start_upper[i] = 0;
            }
            else if (i == d + mpi_sync_start_dims)
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


        MPI_CALL(MPI_Type_create_subarray(ndims,
                                          dims,
                                          s_count_upper,
                                          s_start_upper,
                                          MPI_ORDER_C,
                                          ele_type,
                                          &((*updater)->send_types[2 * d + 0])));

        MPI_CALL(MPI_Type_create_subarray(ndims,
                                          dims,
                                          s_count_lower,
                                          s_start_lower,
                                          MPI_ORDER_C,
                                          ele_type,
                                          &((*updater)->send_types[2 * d + 1])));

        MPI_CALL(MPI_Type_create_subarray(ndims,
                                          dims,
                                          r_count_lower,
                                          r_start_lower,
                                          MPI_ORDER_C,
                                          ele_type,
                                          &((*updater)->recv_types[2 * d + 0])));
        MPI_CALL(MPI_Type_create_subarray(ndims,
                                          dims,
                                          r_count_upper,
                                          r_start_upper,
                                          MPI_ORDER_C,
                                          ele_type,
                                          &((*updater)->recv_types[2 * d + 1])));

        MPI_CALL(MPI_Type_commit(&((*updater)->send_types[2 * d + 0])));
        MPI_CALL(MPI_Type_commit(&((*updater)->send_types[2 * d + 1])));
        MPI_CALL(MPI_Type_commit(&((*updater)->recv_types[2 * d + 0])));
        MPI_CALL(MPI_Type_commit(&((*updater)->recv_types[2 * d + 1])));

        (*updater)->send_displs[2 * d + 0] = 0;
        (*updater)->send_displs[2 * d + 1] = 0;
        (*updater)->recv_displs[2 * d + 0] = 0;
        (*updater)->recv_displs[2 * d + 1] = 0;
    }
}

int spParallelUpdaterCreateIndexed(spMPICartUpdater **updater,
                                   const spDataType *ele_data_desc,
                                   int ndims,
                                   size_type const *num_of_ele,
                                   size_type const **index)
{
    *updater = malloc(sizeof(spMPICartUpdater));

    (*updater)->comm = spMPIComm();

    if ((*updater)->comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }

    int topo_type = MPI_CART;

    MPI_CALL(MPI_Topo_test((*updater)->comm, &topo_type));

    assert(topo_type == MPI_CART);


    MPI_Datatype ele_type = spDataTypeMPIType(ele_data_desc);

    if (ele_type == MPI_DATATYPE_NULL) { return SP_FAILED; }

    int mpi_topology_ndims = 0;

    MPI_CALL(MPI_Cartdim_get((*updater)->comm, &mpi_topology_ndims));

    assert(mpi_topology_ndims <= ndims);


    MPI_Count ele_size_in_byte = 0;

    MPI_CALL(MPI_Type_size_x(ele_type, &ele_size_in_byte));
}

int spMPICartUpdateNdArrayHalo(int num_of_buffer, void **buffers, const spDataType *data_desc, int ndims,
                               const size_type *shape, const size_type *start, const size_type *stride,
                               const size_type *count, const size_type *block, int mpi_sync_start_dims)
{
    spMPICartUpdater *updater;

    SP_CALL(spMPICartUpdaterCreateDA(&updater,
                                     data_desc,
                                     ndims,
                                     shape,
                                     start,
                                     stride,
                                     count,
                                     block,
                                     mpi_sync_start_dims));

    SP_CALL(spMPICartUpdateAll(updater, num_of_buffer, buffers));

    SP_CALL(spMPICartUpdaterDestroy(&updater));
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

    MPI_CALL(MPI_Cartdim_get(comm, &mpi_topology_ndims));

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

size_type spMPISum(size_type v)
{
    UNIMPLEMENTED;
    return v;
}

size_type spMPIPrefixSums(size_type v)
{
    MPI_Comm comm = spMPIComm();

    if (comm == MPI_COMM_NULL) { return SP_FAILED; }
    UNIMPLEMENTED;
}

int spMPIScan(size_type *v, size_type num)
{
    MPI_Comm comm = spMPIComm();

    if (comm == MPI_COMM_NULL) { return SP_FAILED; }

    size_type res[num];

    MPI_CALL(MPI_Scan(v, res, num, MPI_INT64_T, MPI_SUM, comm));

    for (int i = 0; i < num; ++i) { v[i] = res[i] - v[i]; }

    return SP_SUCCESS;
};
