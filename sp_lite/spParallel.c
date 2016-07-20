//
// Created by salmon on 16-7-20.
//
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

void spMPIDataTypeCreate(int count, int *array_of_displacements, int type_tag, MPI_Datatype *new_type)
{
    MPI_Datatype ele_type = MPI_BYTE;
    switch (type_tag)
    {
        case SP_TYPE_float:
            ele_type = MPI_FLOAT;
            break;
        case SP_TYPE_double:
            ele_type = MPI_DOUBLE;
            break;

        case SP_TYPE_int:
            ele_type = MPI_INT;
            break;

        case SP_TYPE_long:
            ele_type = MPI_LONG;
            break;
        case SP_TYPE_int64_t:
            ele_type = MPI_INT64_T;
            break;
        default:
            break;
    }

//    MPI_ERROR(MPI_Type_create_indexed_block(count,
//                                            SP_NUMBER_OF_ENTITIES_IN_PAGE,
//                                            array_of_displacements,
//                                            ele_type,
//                                            new_type));
    MPI_ERROR(MPI_Type_commit(new_type));
}
