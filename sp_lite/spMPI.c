//
// Created by salmon on 16-9-12.
//
#include <assert.h>
#include <mpi.h>

#include "spMPI.h"
#include "spDataType.h"
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

int spMPIUpdateNdArrayHalo2(int num_of_buffer, void **buffers, const spDataType *data_desc, int ndims,
                            const int *shape, const int *start, const int *stride,
                            const int *count, const int *block, int mpi_sync_start_dims)
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

#define SP_MPI_MAX_NUM_NEIGHBOUR 27

typedef struct spMPIUpdater_s
{
    MPI_Comm comm;
    int num_of_neighbour;
    int mpi_send_recv_count[SP_MPI_MAX_NUM_NEIGHBOUR];
    MPI_Datatype send_types[SP_MPI_MAX_NUM_NEIGHBOUR];
    MPI_Datatype recv_types[SP_MPI_MAX_NUM_NEIGHBOUR];
    MPI_Aint send_displs[SP_MPI_MAX_NUM_NEIGHBOUR];
    MPI_Aint recv_displs[SP_MPI_MAX_NUM_NEIGHBOUR];
} spMPIUpdater;

int spMPIUpdaterDestroy(spMPIUpdater **updater)
{
    for (int i = 0; i < (*updater)->num_of_neighbour; ++i)
    {
        if ((*updater)->send_types[i] != MPI_DATATYPE_NULL) MPI_CALL(MPI_Type_free(&((*updater)->send_types[i])));

        if ((*updater)->recv_types[i] != MPI_DATATYPE_NULL) MPI_CALL(MPI_Type_free(&((*updater)->recv_types[i])));
    }

    free(*updater);

}

int spMPIUpdate(spMPIUpdater const *updater, void *buffer)
{


    SP_CALL(spMPINeighborAllToAll(buffer,
                                  updater->mpi_send_recv_count,
                                  updater->send_displs,
                                  updater->send_types,
                                  buffer,
                                  updater->mpi_send_recv_count,
                                  updater->recv_displs,
                                  updater->recv_types,
                                  updater->comm));

    return SP_SUCCESS;

}

int spMPIUpdateAll(spMPIUpdater const *updater, int num_of_buffer, void **buffers)
{
    for (int i = 0; i < num_of_buffer; ++i) {SP_CALL(spMPIUpdate(updater, buffers[i])); }
    return SP_SUCCESS;
}

int spMPIUpdaterCreateDistArray(spMPIUpdater **updater,
                                MPI_Comm comm,
                                const spDataType *old_data,
                                int ndims,
                                const int *shape,
                                const int *start,
                                const int *stride,
                                const int *count,
                                const int *block,
                                int mpi_sync_start_dims)
{
    if (comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }

    *updater = malloc(sizeof(spMPIUpdater));

    (*updater)->comm = comm;

    int topo_type = MPI_CART;

    MPI_CALL(MPI_Topo_test((*updater)->comm, &topo_type));

    assert(topo_type == MPI_CART);


    MPI_Datatype ele_type = spDataTypeMPIType(old_data);

    if (ele_type == MPI_DATATYPE_NULL) { return SP_FAILED; }

    int mpi_topology_ndims = 0;

    MPI_CALL(MPI_Cartdim_get((*updater)->comm, &mpi_topology_ndims));

    assert(mpi_topology_ndims <= ndims);


    MPI_Count ele_size_in_byte = 0;

    MPI_CALL(MPI_Type_size_x(ele_type, &ele_size_in_byte));

    int dims[ndims];

    for (int i = 0; i < ndims; ++i) { dims[i] = shape[i]; }

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

        (*updater)->mpi_send_recv_count[2 * d] = (*updater)->mpi_send_recv_count[2 * d + 1] = 1;

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
                s_count_lower[i] = start[i];
                s_start_lower[i] = start[i];
                s_count_upper[i] = (dims[i] - count[i] - start[i]);
                s_start_upper[i] = (start[i] + count[i] - s_count_upper[i]);

                r_count_lower[i] = start[i];
                r_start_lower[i] = 0;
                r_count_upper[i] = (dims[i] - count[i] - start[i]);
                r_start_upper[i] = dims[i] - s_count_upper[i];
            }
            else
            {
                s_count_lower[i] = count[i];
                s_start_lower[i] = start[i];
                s_count_upper[i] = count[i];
                s_start_upper[i] = start[i];

                r_count_lower[i] = count[i];
                r_start_lower[i] = start[i];
                r_count_upper[i] = count[i];
                r_start_upper[i] = start[i];
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

int spMPIUpdateNdArrayHalo(int num_of_buffer, void **buffers, const spDataType *data_desc,
                           int ndims,
                           const int *shape,
                           const int *start,
                           const int *stride,
                           const int *count,
                           const int *block, int mpi_sync_start_dims)
{
    spMPIUpdater *updater;

    SP_CALL(spMPIUpdaterCreateDistArray(&updater,
                                        spMPIComm(),
                                        data_desc,
                                        ndims,
                                        shape,
                                        start,
                                        stride,
                                        count,
                                        block,
                                        mpi_sync_start_dims));

    SP_CALL(spMPIUpdateAll(updater, num_of_buffer, buffers));

    SP_CALL(spMPIUpdaterDestroy(&updater));
}
int spMPIUpdaterCreateDistIndexed(spMPIUpdater **updater,
                                  MPI_Comm comm,
                                  const spDataType *old_data,
                                  int num_of_dims,
                                  int block_length,
                                  int const *send_count,
                                  MPI_Aint **send_index,
                                  int *send_disp,
                                  int const *recv_count,
                                  MPI_Aint **recv_index,
                                  int *recv_disp)
{

    if (comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }

    *updater = malloc(sizeof(spMPIUpdater));

    (*updater)->comm = comm;

    int topo_type = MPI_CART;

    MPI_CALL(MPI_Topo_test((*updater)->comm, &topo_type));

    assert(topo_type == MPI_CART);


    MPI_Datatype ele_type = spDataTypeMPIType(old_data);

    if (ele_type == MPI_DATATYPE_NULL) { return SP_FAILED; }

    int mpi_topology_ndims = 0;

    MPI_CALL(MPI_Cartdim_get((*updater)->comm, &mpi_topology_ndims));

    assert(mpi_topology_ndims <= num_of_dims);


    MPI_Count ele_size_in_byte = 0;

    MPI_CALL(MPI_Type_size_x(ele_type, &ele_size_in_byte));


    (*updater)->num_of_neighbour = mpi_topology_ndims * 2;

    assert((*updater)->num_of_neighbour <= num_of_dims * 2);

    for (int d = 0; d < mpi_topology_ndims; ++d)
    {

        (*updater)->send_types[2 * d + 0] = MPI_DATATYPE_NULL;
        (*updater)->send_types[2 * d + 1] = MPI_DATATYPE_NULL;
        (*updater)->recv_types[2 * d + 0] = MPI_DATATYPE_NULL;
        (*updater)->recv_types[2 * d + 1] = MPI_DATATYPE_NULL;


        (*updater)->mpi_send_recv_count[2 * d] = (*updater)->mpi_send_recv_count[2 * d + 1] = 1;


        MPI_CALL(MPI_Type_create_hindexed_block(send_count[2 * d + 0],
                                                block_length,
                                                send_index[2 * d + 0],
                                                ele_type,
                                                &((*updater)->send_types[2 * d + 1])));

        MPI_CALL(MPI_Type_create_hindexed_block(send_count[2 * d + 1],
                                                block_length,
                                                send_index[2 * d + 1],
                                                ele_type,
                                                &((*updater)->send_types[2 * d + 1])));


        MPI_CALL(MPI_Type_create_hindexed_block(recv_count[2 * d + 0],
                                                block_length,
                                                recv_index[2 * d + 0],
                                                ele_type,
                                                &((*updater)->send_types[2 * d + 1])));

        MPI_CALL(MPI_Type_create_hindexed_block(recv_count[2 * d + 1],
                                                block_length,
                                                recv_index[2 * d + 1],
                                                ele_type,
                                                &((*updater)->send_types[2 * d + 1])));


        MPI_CALL(MPI_Type_commit(&((*updater)->send_types[2 * d + 0])));
        MPI_CALL(MPI_Type_commit(&((*updater)->send_types[2 * d + 1])));
        MPI_CALL(MPI_Type_commit(&((*updater)->recv_types[2 * d + 0])));
        MPI_CALL(MPI_Type_commit(&((*updater)->recv_types[2 * d + 1])));

        (*updater)->send_displs[2 * d + 0] = (send_disp == NULL) ? 0 : send_disp[2 * d + 0];
        (*updater)->send_displs[2 * d + 1] = (send_disp == NULL) ? 0 : send_disp[2 * d + 1];
        (*updater)->recv_displs[2 * d + 0] = (recv_disp == NULL) ? 0 : recv_disp[2 * d + 0];
        (*updater)->recv_displs[2 * d + 1] = (recv_disp == NULL) ? 0 : recv_disp[2 * d + 1];
    }

}

int spMPISum(int v)
{
    UNIMPLEMENTED;
    return v;
}

int spMPIPrefixSums(int v)
{
    MPI_Comm comm = spMPIComm();

    if (comm == MPI_COMM_NULL) { return SP_FAILED; }
    UNIMPLEMENTED;
}

int spParallelScan(int *v, int num)
{
    MPI_Comm comm = spMPIComm();

    if (comm == MPI_COMM_NULL) { return SP_FAILED; }

    int res[num];

    MPI_CALL(MPI_Scan(v, res, num, MPI_INT64_T, MPI_SUM, comm));

    for (int i = 0; i < num; ++i) { v[i] = res[i] - v[i]; }

    return SP_SUCCESS;
};


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
//    int m_dims_[2] = {5, 5};
//    int start[2] = {1, 1};
//    int count[2] = {3, 3};
//
//    spMPIUpdateNdArrayHalo(buffer, 2, m_dims_, start, NULL, count, NULL, MPI_INT);
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