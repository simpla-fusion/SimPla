//
// Created by salmon on 16-9-12.
//
#include <assert.h>
#include "spMPI.h"
#include "sp_lite_def.h"
#include "spDataType.h"
#include "spObject.h"
#include "spAlogorithm.h"
#include "spParallel.h"

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


//
//int spMPICartUpdateNdArrayHalo2(int num_of_buffer, void **buffers, const spDataType *data_desc, int ndims,
//                                const size_type *shape, const size_type *start, const size_type *stride,
//                                const size_type *count, const size_type *block, int mpi_sync_start_dims)
//{
//    int error_code = SP_SUCCESS;
//
//    MPI_Comm comm = spMPIComm();
//
//    if (comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }
//
//    int tope_type = MPI_CART;
//
//    MPI_CALL(MPI_Topo_test(comm, &tope_type));
//
//    assert(tope_type == MPI_CART);
//
//
//    MPI_Datatype ele_type = spDataTypeMPIType(data_desc);
//
//    if (ele_type == MPI_DATATYPE_NULL) { return SP_FAILED; }
//
//    int mpi_topology_ndims = 0;
//
//    MPI_CALL(MPI_Cartdim_get(comm, &mpi_topology_ndims));
//
//    assert(mpi_topology_ndims <= ndims);
//
//    int num_of_neighbour = 2 * mpi_topology_ndims;
//
//    int mpi_sendrecv_count[num_of_neighbour];
//    MPI_Datatype send_types[num_of_neighbour];
//    MPI_Datatype recv_types[num_of_neighbour];
//    MPI_Aint send_displs[num_of_neighbour];
//    MPI_Aint recv_displs[num_of_neighbour];
//    MPI_Count ele_size_in_byte = 0;
//
//    MPI_CALL(MPI_Type_size_x(ele_type, &ele_size_in_byte));
//
//    int dims[ndims];
//
//    for (int i = 0; i < ndims; ++i) { dims[i] = (int) shape[i]; }
//
//    int s_count_lower[ndims];
//    int s_start_lower[ndims];
//    int s_count_upper[ndims];
//    int s_start_upper[ndims];
//
//    int r_count_lower[ndims];
//    int r_start_lower[ndims];
//    int r_count_upper[ndims];
//    int r_start_upper[ndims];
//
//    for (int d = 0; d < mpi_topology_ndims; ++d)
//    {
//
//        send_types[2 * d + 0] = MPI_DATATYPE_NULL;
//        send_types[2 * d + 1] = MPI_DATATYPE_NULL;
//        recv_types[2 * d + 0] = MPI_DATATYPE_NULL;
//        recv_types[2 * d + 1] = MPI_DATATYPE_NULL;
//
//        if (dims[d] == 1) { continue; }
//
//        mpi_sendrecv_count[2 * d] = mpi_sendrecv_count[2 * d + 1] = 1;
//
//        for (int i = 0; i < ndims; ++i)
//        {
//            if (i >= mpi_sync_start_dims && i < d + mpi_sync_start_dims)
//            {
//                s_count_lower[i] = dims[i];
//                s_start_lower[i] = 0;
//                s_count_upper[i] = dims[i];
//                s_start_upper[i] = 0;
//
//                r_count_lower[i] = dims[i];
//                r_start_lower[i] = 0;
//                r_count_upper[i] = dims[i];
//                r_start_upper[i] = 0;
//            }
//            else if (i == d + mpi_sync_start_dims)
//            {
//                s_count_lower[i] = (int) start[i];
//                s_start_lower[i] = (int) start[i];
//                s_count_upper[i] = (int) (dims[i] - count[i] - start[i]);
//                s_start_upper[i] = (int) (start[i] + count[i] - s_count_upper[i]);
//
//                r_count_lower[i] = (int) start[i];
//                r_start_lower[i] = (int) 0;
//                r_count_upper[i] = (int) (dims[i] - count[i] - start[i]);
//                r_start_upper[i] = (int) dims[i] - s_count_upper[i];
//            }
//            else
//            {
//                s_count_lower[i] = (int) count[i];
//                s_start_lower[i] = (int) start[i];
//                s_count_upper[i] = (int) count[i];
//                s_start_upper[i] = (int) start[i];
//
//                r_count_lower[i] = (int) count[i];
//                r_start_lower[i] = (int) start[i];
//                r_count_upper[i] = (int) count[i];
//                r_start_upper[i] = (int) start[i];
//            };
//        }
//
//
//        MPI_CALL(MPI_Type_create_subarray(ndims,
//                                          dims,
//                                          s_count_upper,
//                                          s_start_upper,
//                                          MPI_ORDER_C,
//                                          ele_type,
//                                          &send_types[2 * d + 0]));
//
//        MPI_CALL(MPI_Type_create_subarray(ndims,
//                                          dims,
//                                          s_count_lower,
//                                          s_start_lower,
//                                          MPI_ORDER_C,
//                                          ele_type,
//                                          &send_types[2 * d + 1]));
//
//        MPI_CALL(MPI_Type_create_subarray(ndims,
//                                          dims,
//                                          r_count_lower,
//                                          r_start_lower,
//                                          MPI_ORDER_C,
//                                          ele_type,
//                                          &recv_types[2 * d + 0]));
//        MPI_CALL(MPI_Type_create_subarray(ndims,
//                                          dims,
//                                          r_count_upper,
//                                          r_start_upper,
//                                          MPI_ORDER_C,
//                                          ele_type,
//                                          &recv_types[2 * d + 1]));
//
//        MPI_CALL(MPI_Type_commit(&(send_types[2 * d + 0])));
//        MPI_CALL(MPI_Type_commit(&(send_types[2 * d + 1])));
//        MPI_CALL(MPI_Type_commit(&(recv_types[2 * d + 0])));
//        MPI_CALL(MPI_Type_commit(&(recv_types[2 * d + 1])));
//
//        send_displs[2 * d + 0] = 0;
//        send_displs[2 * d + 1] = 0;
//        recv_displs[2 * d + 0] = 0;
//        recv_displs[2 * d + 1] = 0;
//    }
//
//    for (int i = 0; i < num_of_buffer; ++i)
//    {
//        SP_CALL(spMPINeighborAllToAll(buffers[i], mpi_sendrecv_count, send_displs, send_types,
//                                      buffers[i], mpi_sendrecv_count, recv_displs, recv_types, comm));
//    }
//
//
//    for (int i = 0; i < num_of_neighbour; ++i)
//    {
//        if (send_types[i] != MPI_DATATYPE_NULL) MPI_CALL(MPI_Type_free(&send_types[i]));
//
//        if (recv_types[i] != MPI_DATATYPE_NULL) MPI_CALL(MPI_Type_free(&recv_types[i]));
//    }
//
//    return error_code;
//
//}

MPI_Datatype spMPIDataType(int type_tag)
{
    MPI_Datatype res_type = MPI_DATATYPE_NULL;
    switch (type_tag)
    {
        case SP_TYPE_int:
            res_type = MPI_INT;
            break;
        case SP_TYPE_long:
            res_type = MPI_LONG;
            break;
        case SP_TYPE_unsigned_int:
            res_type = MPI_UNSIGNED;
            break;
        case SP_TYPE_unsigned_long:
            res_type = MPI_UNSIGNED_LONG;
            break;
        case SP_TYPE_float:
            res_type = MPI_FLOAT;
            break;
        case SP_TYPE_double:
            res_type = MPI_DOUBLE;
            break;
//    case SP_TYPE_std::complex<double>:     res_type = MPI_2DOUBLE_COMPLEX; break;
//    case SP_TYPE_std::complex<float>:      res_type = MPI_2COMPLEX; break;
        default:
            UNIMPLEMENTED;
            break;

    }
    return res_type;
}

int spMPIUpdaterCreate(spMPIUpdater **updater, size_type size)
{

    *updater = (spMPIUpdater *) malloc(size);
    (*updater)->comm = spMPIComm();

    if ((*updater)->comm == MPI_COMM_NULL) { return SP_FAILED; }

    (*updater)->num_of_neighbour = 6;

    for (int i = 0; i < (*updater)->num_of_neighbour; ++i)
    {
        (*updater)->send_types[i] = MPI_DATATYPE_NULL;
        (*updater)->recv_types[i] = MPI_DATATYPE_NULL;

        (*updater)->send_count[i] = 0;
        (*updater)->recv_count[i] = 0;

        (*updater)->send_displs[i] = 0;
        (*updater)->recv_displs[i] = 0;


        (*updater)->send_buffer[i] = NULL;
        (*updater)->recv_buffer[i] = NULL;
    }
    return SP_SUCCESS;
};

int spMPIUpdaterDestroy(spMPIUpdater **updater)
{
    for (int i = 0; i < (*updater)->num_of_neighbour; ++i)
    {
        if ((*updater)->send_types[i] != MPI_DATATYPE_NULL) MPI_CALL(MPI_Type_free(&((*updater)->send_types[i])));
        if ((*updater)->recv_types[i] != MPI_DATATYPE_NULL) MPI_CALL(MPI_Type_free(&((*updater)->recv_types[i])));

        SP_CALL(spMemDeviceFree(&((*updater)->send_buffer[i])));
        SP_CALL(spMemDeviceFree(&((*updater)->recv_buffer[i])));
    }
    free(*updater);
    *updater = NULL;
    return SP_SUCCESS;
};

int spMPIPrefixSum(size_type *p_offset, size_type *p_count)
{
    MPI_Comm comm = spMPIComm();

    if (comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }

    int num_of_process = spMPISize();

    int process_num = spMPIRank();

    int offset = (p_offset != NULL) ? (int) *p_offset : 0;
    int count = (p_count != NULL) ? (int) *p_count : 1;

    int buffer[num_of_process + 1];


    MPI_CALL(MPI_Barrier(comm));

    MPI_CALL(MPI_Gather(&count, 1, MPI_INT, &buffer[0], 1, MPI_INT, 0, comm));

    MPI_CALL(MPI_Barrier(comm));

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

    MPI_CALL(MPI_Barrier(comm));
    MPI_CALL(MPI_Scatter(&buffer[0], 1, MPI_INT, &offset, 1, MPI_INT, 0, comm));
    MPI_CALL(MPI_Barrier(comm));
    MPI_CALL(MPI_Bcast(&count, 1, MPI_INT, 0, comm));
    MPI_CALL(MPI_Barrier(comm));

    if (p_count != NULL) { *p_count = (size_type) count; }
    if (p_offset != NULL) { *p_offset = (size_type) offset; }

    return SP_SUCCESS;
}


#define      spMPIHaloUpdater_s_header                   \
    SP_MPI_UPDATER_HEAD                                  \
    size_type dims[SP_MAX_NUM_DIMS];                     \
    size_type strides[SP_MAX_NUM_DIMS];                  \
    size_type s_count_lower[3][SP_MAX_NUM_DIMS];         \
    size_type s_start_lower[3][SP_MAX_NUM_DIMS];         \
    size_type s_count_upper[3][SP_MAX_NUM_DIMS];         \
    size_type s_start_upper[3][SP_MAX_NUM_DIMS];         \
                                                         \
    size_type r_count_lower[3][SP_MAX_NUM_DIMS];         \
    size_type r_start_lower[3][SP_MAX_NUM_DIMS];         \
    size_type r_count_upper[3][SP_MAX_NUM_DIMS];         \
    size_type r_start_upper[3][SP_MAX_NUM_DIMS];         \


typedef struct spMPIHaloUpdater_s
{
    spMPIHaloUpdater_s_header
};

int spMPIHaloUpdaterDestroy(spMPIHaloUpdater **updater)
{
    SP_CALL(spMPIUpdaterDestroy((spMPIUpdater **) (updater)));
    return SP_SUCCESS;
}

int spMPIHaloUpdaterCreate(spMPIHaloUpdater **updater, int data_type_tag)
{
    SP_CALL(spMPIUpdaterCreate((spMPIUpdater **) updater, sizeof(spMPIHaloUpdater)));

    MPI_Datatype ele_type = spMPIDataType(data_type_tag);

    if (ele_type == MPI_DATATYPE_NULL) { return SP_FAILED; }

    for (int d = 0; d < 3; ++d)
    {

        (*updater)->send_types[2 * d + 0] = ele_type;
        (*updater)->send_types[2 * d + 1] = ele_type;
        (*updater)->recv_types[2 * d + 0] = ele_type;
        (*updater)->recv_types[2 * d + 1] = ele_type;
    }

    return SP_SUCCESS;

}

int spMPIHaloUpdaterDeploy(spMPIHaloUpdater *updater,
                           int mpi_sync_start_dims, int ndims,
                           const size_type *shape, const size_type *start, const size_type *stride,
                           const size_type *count, const size_type *block)
{


    int topo_type = MPI_CART;

    MPI_CALL(MPI_Topo_test(updater->comm, &topo_type));

    assert(topo_type == MPI_CART);


    int mpi_topology_ndims = 0;

    MPI_CALL(MPI_Cartdim_get(updater->comm, &mpi_topology_ndims));

    assert(mpi_topology_ndims <= ndims);

    for (int i = 0; i < ndims; ++i) { updater->dims[i] = shape[i]; }

    updater->num_of_neighbour = mpi_topology_ndims * 2;

    for (int d = 0; d < mpi_topology_ndims; ++d)
    {
        updater->send_count[2 * d + 0] = 1;
        updater->send_count[2 * d + 1] = 1;
        updater->recv_count[2 * d + 0] = 1;
        updater->recv_count[2 * d + 1] = 1;


        updater->send_displs[2 * d + 0] = 0;
        updater->send_displs[2 * d + 1] = 0;
        updater->recv_displs[2 * d + 0] = 0;
        updater->recv_displs[2 * d + 1] = 0;


        if (updater->dims[d] == 1) { continue; }

        for (int i = 0; i < ndims; ++i)
        {
            if (i >= mpi_sync_start_dims && i < d + mpi_sync_start_dims)
            {
                updater->s_count_lower[d][i] = updater->dims[i];
                updater->s_start_lower[d][i] = 0;
                updater->s_count_upper[d][i] = updater->dims[i];
                updater->s_start_upper[d][i] = 0;

                updater->r_count_lower[d][i] = updater->dims[i];
                updater->r_start_lower[d][i] = 0;
                updater->r_count_upper[d][i] = updater->dims[i];
                updater->r_start_upper[d][i] = 0;
            } else if (i == d + mpi_sync_start_dims)
            {
                updater->s_count_lower[d][i] = start[i];
                updater->s_start_lower[d][i] = start[i];
                updater->s_count_upper[d][i] = (updater->dims[i] - count[i] - start[i]);
                updater->s_start_upper[d][i] = (start[i] + count[i] - updater->s_count_upper[d][i]);

                updater->r_count_lower[d][i] = start[i];
                updater->r_start_lower[d][i] = 0;
                updater->r_count_upper[d][i] = (updater->dims[i] - count[i] - start[i]);
                updater->r_start_upper[d][i] = updater->dims[i] - updater->s_count_upper[d][i];
            } else
            {
                updater->s_count_lower[d][i] = count[i];
                updater->s_start_lower[d][i] = start[i];
                updater->s_count_upper[d][i] = count[i];
                updater->s_start_upper[d][i] = start[i];

                updater->r_count_lower[d][i] = count[i];
                updater->r_start_lower[d][i] = start[i];
                updater->r_count_upper[d][i] = count[i];
                updater->r_start_upper[d][i] = start[i];
            };


            updater->send_count[2 * d + 0] *= updater->s_count_lower[d][i];
            updater->send_count[2 * d + 1] *= updater->s_count_upper[d][i];
            updater->recv_count[2 * d + 0] *= updater->r_count_lower[d][i];
            updater->recv_count[2 * d + 1] *= updater->r_count_upper[d][i];
        }
    }
    updater->strides[ndims - 1] = 1;
    for (int i = ndims - 2; i >= 0; ++i)
    {
        updater->strides[i] = updater->dims[i - 1] * updater->strides[i - 1];

    }
    return SP_SUCCESS;
}

int spMPIHaloUpdate(spMPIHaloUpdater *updater, void *data)
{

    if (updater->comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }

    int tope_type = MPI_CART;

    MPI_CALL(MPI_Topo_test(updater->comm, &tope_type));

    if (tope_type != MPI_CART) { return SP_FAILED; }

    int tag = 0;

    int mpi_topology_ndims = 0;

    MPI_CALL(MPI_Cartdim_get(updater->comm, &mpi_topology_ndims));

    for (int d = 0; d < mpi_topology_ndims; ++d)
    {
        if (updater->send_types[d * 2 + 0] == MPI_DATATYPE_NULL) { continue; }

        SP_CALL(spMemoryCopySubArray(updater->send_buffer[2 * d + 0], data,
                                     updater->strides,
                                     updater->s_start_lower[d],
                                     updater->s_count_lower[d]));

        SP_CALL(spMemoryCopySubArray(updater->send_buffer[2 * d + 1], data,
                                     updater->strides,
                                     updater->s_start_lower[d],
                                     updater->s_count_lower[d]));


        int left, right;

        MPI_CALL(MPI_Cart_shift(updater->comm, d, 1, &left, &right));

        MPI_CALL(MPI_Sendrecv(
                (byte_type *) (updater->send_buffer[d * 2 + 0]) + updater->send_displs[d * 2 + 0],
                updater->send_count[d * 2 + 0],
                updater->send_types[d * 2 + 0],
                left, tag,

                (byte_type *) (updater->recv_buffer[d * 2 + 1]) + updater->recv_displs[d * 2 + 1],
                updater->recv_count[d * 2 + 1],
                updater->recv_types[d * 2 + 1],
                right, tag,

                updater->comm,
                MPI_STATUS_IGNORE));

        MPI_CALL(MPI_Sendrecv(
                (byte_type *) (updater->send_buffer[d * 2 + 1]) + updater->send_displs[d * 2 + 1],
                updater->send_count[d * 2 + 1],
                updater->send_types[d * 2 + 1],
                right, tag,

                (byte_type *) (updater->recv_buffer[d * 2 + 0]) + updater->recv_displs[d * 2 + 0],
                updater->recv_count[d * 2 + 0],
                updater->recv_types[d * 2 + 0],
                left, tag,
                updater->comm,
                MPI_STATUS_IGNORE));

        SP_CALL(spMemoryCopyInvSubArray(data, updater->send_buffer[2 * d + 0],
                                        updater->strides,
                                        updater->r_start_lower[d],
                                        updater->r_count_lower[d]));

        SP_CALL(spMemoryCopyInvSubArray(data, updater->send_buffer[2 * d + 1],
                                        updater->strides,
                                        updater->r_start_lower[d],
                                        updater->r_count_lower[d]));
    }
    return SP_SUCCESS;

}

int spMPIHaloUpdateAll(spMPIHaloUpdater *updater, int num_of_buffer, void **buffers)
{
    for (int i = 0; i < num_of_buffer; ++i) {SP_CALL(spMPIHaloUpdate(updater, buffers[i])); }
    return SP_SUCCESS;
}

struct spMPINoncontiguousUpdater_s
{
    spMPIHaloUpdater_s_header

    size_type *send_index[6];
    size_type *recv_index[6];
};

int spMPINoncontiguousUpdaterCreate(spMPINoncontiguousUpdater **updater, int data_type_tag)
{
    SP_CALL(spMPIUpdaterCreate((spMPIUpdater **) updater, sizeof(spMPINoncontiguousUpdater)));

    MPI_Datatype ele_type = spMPIDataType(data_type_tag);

    if (ele_type == MPI_DATATYPE_NULL) { return SP_FAILED; }

    for (int i = 0; i < 6; ++i)
    {
        (*updater)->send_index[i] = NULL;
        (*updater)->recv_index[i] = NULL;
        (*updater)->send_types[i] = ele_type;
        (*updater)->recv_types[i] = ele_type;
    }
    return SP_SUCCESS;
}


int spMPINoncontiguousUpdaterDestroy(spMPINoncontiguousUpdater **updater)
{

    for (int i = 0; i < 6; ++i)
    {
        SP_CALL(spMemDeviceFree((void **) &((*updater)->send_index[i])));
        SP_CALL(spMemDeviceFree((void **) &((*updater)->recv_index[i])));
    }
    SP_CALL(spMPIUpdaterDestroy((spMPIUpdater **) (updater)));

    return SP_SUCCESS;
}

int spMPINoncontiguousUpdate(spMPINoncontiguousUpdater *updater, void *data)
{

    if (updater->comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }

    int tope_type = MPI_CART;

    MPI_CALL(MPI_Topo_test(updater->comm, &tope_type));

    if (tope_type != MPI_CART) { return SP_FAILED; }

    int tag = 0;

    int mpi_topology_ndims = 0;

    MPI_CALL(MPI_Cartdim_get(updater->comm, &mpi_topology_ndims));

    for (int d = 0; d < mpi_topology_ndims; ++d)
    {
        if (updater->send_types[d * 2 + 0] == MPI_DATATYPE_NULL) { continue; }

        SP_CALL(spMemoryCopyIndirect(updater->send_buffer[2 * d + 0], data,
                                     (size_type) updater->send_count[2 * d + 0],
                                     updater->send_index[2 * d + 0]));

        SP_CALL(spMemoryCopyIndirect(updater->send_buffer[2 * d + 1], data,
                                     (size_type) updater->send_count[2 * d + 1],
                                     updater->send_index[2 * d + 1]));


        int left, right;

        MPI_CALL(MPI_Cart_shift(updater->comm, d, 1, &left, &right));

        MPI_CALL(MPI_Sendrecv(
                (byte_type *) (updater->send_buffer[d * 2 + 0]) + updater->send_displs[d * 2 + 0],
                updater->send_count[d * 2 + 0],
                updater->send_types[d * 2 + 0],
                left, tag,

                (byte_type *) (updater->recv_buffer[d * 2 + 1]) + updater->recv_displs[d * 2 + 1],
                updater->recv_count[d * 2 + 1],
                updater->recv_types[d * 2 + 1],
                right, tag,

                updater->comm,
                MPI_STATUS_IGNORE));

        MPI_CALL(MPI_Sendrecv(
                (byte_type *) (updater->send_buffer[d * 2 + 1]) + updater->send_displs[d * 2 + 1],
                updater->send_count[d * 2 + 1],
                updater->send_types[d * 2 + 1],
                right, tag,

                (byte_type *) (updater->recv_buffer[d * 2 + 0]) + updater->recv_displs[d * 2 + 0],
                updater->recv_count[d * 2 + 0],
                updater->recv_types[d * 2 + 0],
                left, tag,
                updater->comm,
                MPI_STATUS_IGNORE));

        SP_CALL(spMemoryCopyInvIndirect(data,
                                        updater->recv_buffer[2 * d + 0],
                                        (size_type) updater->recv_count[2 * d + 0],
                                        updater->recv_index[2 * d + 0]));

        SP_CALL(spMemoryCopyInvIndirect(data,
                                        updater->recv_buffer[2 * d + 1],
                                        (size_type) updater->recv_count[2 * d + 1],
                                        updater->recv_index[2 * d + 1]));
    }
    return SP_SUCCESS;

}

int spMPINoncontiguousUpdateAll(spMPINoncontiguousUpdater *updater, int num_of_buffer, void **buffers)
{
    for (int i = 0; i < num_of_buffer; ++i) {SP_CALL(spMPINoncontiguousUpdate(updater, buffers[i])); }
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
//int spMPIUpdate(spMPIUpdater *updater)
//{
//
//    if (updater->comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }
//
//    int tope_type = MPI_CART;
//
//    MPI_CALL(MPI_Topo_test(updater->comm, &tope_type));
//
//    if (tope_type != MPI_CART) { return SP_FAILED; }
//
//    int tag = 0;
//
//    int mpi_topology_ndims = 0;
//
//    MPI_CALL(MPI_Cartdim_get(updater->comm, &mpi_topology_ndims));
//
//    for (int d = 0; d < mpi_topology_ndims; ++d)
//    {
//        if (updater->send_types[d * 2 + 0] == MPI_DATATYPE_NULL) { continue; }
//        int left, right;
//
//        MPI_CALL(MPI_Cart_shift(updater->comm, d, 1, &left, &right));
//
//        MPI_CALL(MPI_Sendrecv(
//                (byte_type *) (updater->send_buffer[d * 2 + 0]) + updater->send_displs[d * 2 + 0],
//                updater->send_count[d * 2 + 0],
//                updater->send_types[d * 2 + 0],
//                left, tag,
//
//                (byte_type *) (updater->recv_buffer[d * 2 + 1]) + updater->recv_displs[d * 2 + 1],
//                updater->recv_count[d * 2 + 1],
//                updater->recv_types[d * 2 + 1],
//                right, tag,
//
//                updater->comm,
//                MPI_STATUS_IGNORE));
//
//        MPI_CALL(MPI_Sendrecv(
//                (byte_type *) (updater->send_buffer[d * 2 + 1]) + updater->send_displs[d * 2 + 1],
//                updater->send_count[d * 2 + 1],
//                updater->send_types[d * 2 + 1],
//                right, tag,
//
//                (byte_type *) (updater->recv_buffer[d * 2 + 0]) + updater->recv_displs[d * 2 + 0],
//                updater->recv_count[d * 2 + 0],
//                updater->recv_types[d * 2 + 0],
//                left, tag,
//                updater->comm,
//                MPI_STATUS_IGNORE));
//    }
//
//
//    return SP_SUCCESS;
//
//}