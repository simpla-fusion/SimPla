//
// Created by salmon on 16-9-24.
//

extern "C"
{
#include "../../sp_lite_def.h"
#include "../../spMPI.h"
#include "../sp_device.h"
}

/**
 *  @todo need parallel opt.
 * @param id_list
 * @param ndims
 * @param dims
 * @param start
 * @param count
 * @param bucket_start
 * @param bucket_count
 * @param sorted_idx
 * @return
 */
int _GatherIndex(int **id_list, const int *strides, const int *start, const int *count,
                 const size_type *bucket_start,
                 const size_type *bucket_count, const size_type *sorted_idx)
{

    int num = 0;
//#pragma omp parallel for
    for (int i = 0; i < count[0]; ++i)
        for (int j = 0; j < count[1]; ++j)
            for (int k = 0; k < count[2]; ++k)
            {
                size_type s = (size_type) (
                        (start[0] + i) * strides[0] +
                        (start[1] + j) * strides[1] +
                        (start[2] + k) * strides[2]);
                num += bucket_count[s];
            }

    *id_list = malloc(num * sizeof(size_type));

    num = 0;
//#pragma omp parallel for
    for (int i = 0; i < count[0]; ++i)
        for (int j = 0; j < count[1]; ++j)
            for (int k = 0; k < count[2]; ++k)
            {
                size_type s = (size_type) (
                        (start[0] + i) * strides[0] +
                        (start[1] + j) * strides[1] +
                        (start[2] + k) * strides[2]);
                for (int l = 0; l < bucket_count[s]; ++l)
                {
                    (*id_list)[num] = (int) (sorted_idx[bucket_start[s] + l]);
                    ++num;
                }

            }
    return num;
}
//int spMPINoncontiguousUpdaterCreate(spMPINoncontiguousUpdater **updater)
//{
//    *updater = (spMPINoncontiguousUpdater *) malloc(sizeof(spMPINoncontiguousUpdater));
//    {
////            assert(data_type_tag == SP_TYPE_Real);
//        int *disp = NULL;
//        int num = 0;
//
//        int strides[3] = {dims[1] * dims[2], dims[2], dims[2] <= 1 ? 0 : 1};
//
//        num = _GatherIndex(&disp, strides, s_start_lower, s_count_lower, bucket_start, bucket_count, sorted_idx);
//        MPI_CALL(MPI_Type_create_indexed_block(num, (num > 0) ? 1 : 0, disp, ele_type,
//                                               &((*updater)->send_types[2 * d + 0])));
//        (*updater)->send_count[2 * d + 0] = (num > 0) ? 1 : 0;
//        MPI_CALL(MPI_Type_commit(&((*updater)->send_types[2 * d + 0])));
//        free(disp);
//
//
//        num = _GatherIndex(&disp, strides, r_start_lower, r_count_lower, bucket_start, bucket_count, sorted_idx);
//        MPI_CALL(MPI_Type_create_indexed_block(num, (num > 0) ? 1 : 0, disp, ele_type,
//                                               &((*updater)->recv_types[2 * d + 0])));
//        (*updater)->recv_count[2 * d + 0] = (num > 0) ? 1 : 0;
//        MPI_CALL(MPI_Type_commit(&((*updater)->recv_types[2 * d + 0])));
//        free(disp);
//
//
//        num = _GatherIndex(&disp, strides, s_start_upper, s_count_upper, bucket_start, bucket_count, sorted_idx);
//        MPI_CALL(MPI_Type_create_indexed_block(num, (num > 0) ? 1 : 0, disp, ele_type,
//                                               &((*updater)->send_types[2 * d + 1])));
//        (*updater)->send_count[2 * d + 1] = (num > 0) ? 1 : 0;
//        MPI_CALL(MPI_Type_commit(&((*updater)->send_types[2 * d + 1])));
//        free(disp);
//
//
//        num = _GatherIndex(&disp, strides, r_start_upper, r_count_upper, bucket_start, bucket_count, sorted_idx);
//        MPI_CALL(MPI_Type_create_indexed_block(num, (num > 0) ? 1 : 0, disp, ele_type,
//                                               &((*updater)->recv_types[2 * d + 1])));
//        (*updater)->recv_count[2 * d + 1] = (num > 0) ? 1 : 0;
//        MPI_CALL(MPI_Type_commit(&((*updater)->recv_types[2 * d + 1])));
//        free(disp);
//
//    }
//    return SP_SUCCESS;
//}
//
//int spMPINoncontiguousUpdaterDestroy(spMPINoncontiguousUpdater **updater)
//{
//
//    free(*updater);
//    *updater = NULL;
//
//    return SP_SUCCESS;
//}