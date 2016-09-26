//
// Created by salmon on 16-9-26.
//


extern "C"
{
#include "../../spField.h"
#include "../../spMesh.h"
#include "../../spDataType.h"
#include "../sp_device.h"
}


__global__
void spFieldFillSeqInt_kernel(size_type *d, dim3 count, dim3 l_min, dim3 l_strides, dim3 g_min, dim3 g_strides)
{
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x < count.x && y < count.y && z < count.z)
    {
        d[(l_min.x + x) * l_strides.x +
          (l_min.y + y) * l_strides.y +
          (l_min.z + z) * l_strides.z
        ] = (g_min.x + x) * g_strides.x +
            (g_min.y + y) * g_strides.y +
            (g_min.z + z) * g_strides.z;
    }
}


__global__
void spFieldFillSeqReal_kernel(Real *d, dim3 count, dim3 l_min, dim3 l_strides, dim3 g_min, dim3 g_strides)
{
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x < count.x && y < count.y && z < count.z)
    {
        d[(l_min.x + x) * l_strides.x +
          (l_min.y + y) * l_strides.y +
          (l_min.z + z) * l_strides.z
        ] = (g_min.x + x) * g_strides.x +
            (g_min.y + y) * g_strides.y +
            (g_min.z + z) * g_strides.z;
    }
}

int spFieldFillSeq(spField *f, int tag)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) f);
    size_type l_min[3], l_count[3], l_strides[3];

    SP_CALL(spMeshGetDomain(m, tag, l_min, NULL, l_count));
    SP_CALL(spMeshGetStrides(m, l_strides));

    size_type g_dims[3], g_strides[3];
    size_type g_min[3];
    SP_CALL(spMeshGetGlobalDims(m, g_dims));
    SP_CALL(spMeshGetGlobalStart(m, g_min));

    g_strides[0] = g_dims[1] * g_dims[2];
    g_strides[1] = g_dims[2];
    g_strides[2] = g_dims[2] > 0 ? 1 : 0;

    int num_of_sub = spFieldNumberOfSub(f);
    void *d[num_of_sub];
    SP_CALL(spFieldSubArray(f, d));

    size_type grid_dim[3] = {l_count[0], l_count[1], l_count[2]}, block_dim[3];

    SP_CALL(spMeshGetGlobalDims(m, grid_dim));

    SP_CALL(spParallelThreadBlockDecompose(256, grid_dim, block_dim));

    for (int i = 0; i < num_of_sub; ++i)
    {
        switch (spFieldDataType(f))
        {
            case SP_TYPE_size_type:

            SP_CALL_DEVICE_KERNEL(spFieldFillSeqInt_kernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                                  (size_type *) d[i],
                                  sizeType2Dim3(l_count),
                                  sizeType2Dim3(l_min),
                                  sizeType2Dim3(l_strides),
                                  sizeType2Dim3(g_min),
                                  sizeType2Dim3(g_strides));
                break;
            case SP_TYPE_Real:

            SP_CALL_DEVICE_KERNEL(spFieldFillSeqReal_kernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                                  (Real *) d[i],
                                  sizeType2Dim3(l_count),
                                  sizeType2Dim3(l_min),
                                  sizeType2Dim3(l_strides),
                                  sizeType2Dim3(g_min),
                                  sizeType2Dim3(g_strides));
                break;
            default:
                UNIMPLEMENTED;
                break;

        }
    }
    return SP_SUCCESS;
}
