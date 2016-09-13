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

int _GatherIndex(int **id_list,
                 int ndims,
                 const int *dims,
                 const int *start,
                 const int *count,
                 const size_type *start_pos,
                 const size_type *end_pos,
                 const size_type *index)
{
    int strides[ndims];
    return 0;
}
int spMPICartUpdaterCreate(spMPICartUpdater **updater,
                           MPI_Comm comm,
                           const spDataType *data_desc,
                           int mpi_sync_start_dims,
                           int ndims,
                           const size_type *shape,
                           const size_type *start,
                           const size_type *stride,
                           const size_type *count,
                           const size_type *block,
                           const size_type *start_pos,
                           const size_type *end_pos,
                           const size_type *index)
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


    int is_structured = (start_pos != NULL && end_pos != NULL && index != NULL) ? SP_TRUE : SP_FALSE;

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
        if (is_structured == SP_TRUE)
        {


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
        }
        else
        {
            int *disp = NULL;
            int num = 0;


            num = _GatherIndex(&disp, 0, dims, s_start_lower, s_count_lower, start_pos, end_pos, index);
            MPI_CALL(MPI_Type_create_indexed_block(num, 1, disp, ele_type, &((*updater)->send_types[2 * d + 0])));
            free(disp);

            num = _GatherIndex(&disp, 0, dims, s_start_upper, s_count_upper, start_pos, end_pos, index);
            MPI_CALL(MPI_Type_create_indexed_block(num, 1, disp, ele_type, &((*updater)->send_types[2 * d + 1])));
            free(disp);

            num = _GatherIndex(&disp, 0, dims, r_start_lower, r_count_lower, start_pos, end_pos, index);
            MPI_CALL(MPI_Type_create_indexed_block(num, 1, disp, ele_type, &((*updater)->recv_types[2 * d + 0])));
            free(disp);

            num = _GatherIndex(&disp, 0, dims, r_start_upper, r_count_upper, start_pos, end_pos, index);
            MPI_CALL(MPI_Type_create_indexed_block(num, 1, disp, ele_type, &((*updater)->recv_types[2 * d + 1])));
            free(disp);

        }


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

//
//int spMPICartUpdateNdArrayHalo(int num_of_buffer, void **buffers,
//                               const spDataType *data_desc,
//                               int ndims,
//                               const size_type *shape,
//                               const size_type *start,
//                               const size_type *stride,
//                               const size_type *count,
//                               const size_type *block)
//{
//    spMPICartUpdater *updater;
//
//    SP_CALL(spMPICartUpdaterCreate(&updater,
//                                     spMPIComm(),
//                                     data_desc,
//                                     0,
//                                     ndims,
//                                     shape,
//                                     start,
//                                     stride,
//                                     count,
//                                     block,
//                                     NULL,
//                                     NULL,
//                                     NULL));
//
//    SP_CALL(spMPICartUpdateAll(updater, num_of_buffer, buffers));
//
//    SP_CALL(spMPICartUpdaterDestroy(&updater));
//}

int spMPIPrefixSum(size_type *start, size_type *total)
{
    assert(sizeof(size_type) == sizeof(long));

    MPI_Comm comm = spMPIComm();

    size_type count = *start;

    if (comm == MPI_COMM_NULL) { return SP_FAILED; }

    int num_of_process = spMPISize();

    int process_num = spMPIRank();

    size_type buffer[num_of_process];


    MPI_Barrier(comm);

    MPI_Gather(&count, 1, MPI_LONG_INT, &buffer[0], 1, MPI_LONG_INT, 0, comm);

    MPI_Barrier(comm);

    if (process_num == 0)
    {
        for (int i = 1; i < num_of_process; ++i)
        {
            buffer[i] += buffer[i - 1];
        }
        buffer[0] = count;
        count = buffer[num_of_process - 1];

        for (int i = num_of_process - 1; i > 0; --i)
        {
            buffer[i] = buffer[i - 1];
        }
        buffer[0] = 0;
    }
    MPI_Barrier(comm);
    MPI_Scatter(&buffer[0], 1, MPI_LONG_INT, start, 1, MPI_LONG_INT, 0, comm);
    MPI_Barrier(comm);
    MPI_Bcast(&count, 1, MPI_LONG_INT, 0, comm);
    MPI_Barrier(comm);

    if (total != NULL) { *total = count; }

    return SP_SUCCESS;
}
//
//int spMPIScan(size_type *v, size_type num)
//{
//    MPI_Comm comm = spMPIComm();
//
//    if (comm == MPI_COMM_NULL) { return SP_FAILED; }
//
//    size_type res[num];
//
//    MPI_CALL(MPI_Scan(v, res, num, MPI_INT64_T, MPI_SUM, comm));
//
//    for (int i = 0; i < num; ++i) { v[i] = res[i] - v[i]; }
//
//    return SP_SUCCESS;
//};
