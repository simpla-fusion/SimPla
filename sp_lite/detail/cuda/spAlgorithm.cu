//
// Created by salmon on 16-9-13.
//
extern "C"
{
#include <assert.h>
#include "../../spAlogorithm.h"
#include "../sp_device.h"

}

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "../../../../../../../usr/local/cuda/include/host_defines.h"
#include "../../../../../../../usr/local/cuda/include/device_launch_parameters.h"

int sort_by_key(size_type const *key_start, size_type const *key_end, size_type *value)
{
    thrust::sort_by_key(thrust::device_ptr<size_type>((size_type *) key_start),
                        thrust::device_ptr<size_type>((size_type *) key_end),
                        thrust::device_ptr<size_type>(value));

    return SP_SUCCESS;
};

__global__
void spMemoryIndirectCopyKernel(Real *dest, Real const *src, size_type num, size_type const *index)
{
    uint s = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (s < num) { dest[s] = src[index[s]]; }
};

int spMemoryCopyIndirect(Real *dest, Real const *src, size_type num, size_type const *index)
{
    /*@formatter:off*/
   spMemoryIndirectCopyKernel<<<num/256+1,256>>>(dest,src,num, index);
    /*@formatter:on*/
    return SP_SUCCESS;
}


__global__
void spMemoryInvIndirectCopyKernel(Real *dest, Real const *src, size_type num, size_type const *index)
{
    uint s = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (s < num) { dest[index[s]] = src[s]; }
};

int spMemoryCopyInvIndirect(Real *dest, Real const *src, size_type num, size_type const *index)
{
    /*@formatter:off*/
   spMemoryInvIndirectCopyKernel<<<num/256+1,256>>>(dest,src,num, index);
    /*@formatter:on*/
    return SP_SUCCESS;
}

__global__
void spMemoryCopySubArrayKernel(Real *dest, Real const *src,
                                dim3 strides,
                                dim3 start,
                                dim3 count)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint z = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    if (x < count.x && y < count.y && z < count.z)
    {
        dest[(x * count.y + y) * count.z + z] = src[x * strides.x + y * strides.y + z * strides.z];
    }

}

int spMemoryCopySubArray(Real *dest, Real const *src,
                         size_type const *strides,
                         size_type const *start,
                         size_type const *count)
{
    size_type grid_dim[3] = {count[0], count[1], count[2]};
    size_type block_dim[3];

    SP_CALL(spParallelThreadBlockDecompose(128, grid_dim, block_dim));

    /*@formatter:off*/
    SP_CALL_DEVICE_KERNEL( spMemoryCopySubArrayKernel,sizeType2Dim3(grid_dim),sizeType2Dim3(block_dim),
                  dest,src, sizeType2Dim3(strides),sizeType2Dim3(start),sizeType2Dim3(count)));
    /*@formatter:on*/
    return SP_SUCCESS;
}

__global__
void spMemoryCopyInvSubArrayKernel(Real *dest, Real const *src,
                                   dim3 strides,
                                   dim3 start,
                                   dim3 count)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint z = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    if (x < count.x && y < count.y && z < count.z)
    {
        dest[x * strides.x + y * strides.y + z * strides.z] = src[(x * count.y + y) * count.z + z];
    }

}

int spMemoryCopyInvSubArray(Real *dest, Real const *src,
                            size_type const *strides,
                            size_type const *start,
                            size_type const *count)
{
    size_type grid_dim[3] = {count[0], count[1], count[2]};
    size_type block_dim[3];

    SP_CALL(spParallelThreadBlockDecompose(128, grid_dim, block_dim));

    /*@formatter:off*/
    SP_CALL_DEVICE_KERNEL( spMemoryCopyInvSubArrayKernel,   sizeType2Dim3(grid_dim),sizeType2Dim3(block_dim),
                  dest,src, sizeType2Dim3(strides),sizeType2Dim3(start),sizeType2Dim3(count)));
    /*@formatter:on*/
    return SP_SUCCESS;


}

__global__
void spFillSeqIntKernel(size_type *v, size_type num, size_type min, size_type step)
{
    uint s = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (s < num) { v[s] = s * step + min; }
};

int spFillSeqInt(size_type *v, size_type num, size_type min, size_type step)
{
    /*@formatter:off*/
   spFillSeqIntKernel<<<num/256+1,256>>>(v,num,min,step);
    /*@formatter:on*/
    return SP_SUCCESS;
};

__global__
void spTransformMinusKernel(size_type *v, size_type const *a, size_type const *b, size_type num)
{
    uint s = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (s < num) { v[s] = a[s] - b[s]; }
};

int spTransformMinus(size_type *v, size_type const *a, size_type const *b, size_type num)
{
    /*@formatter:off*/
   spTransformMinusKernel<<<num/256+1,256>>>(v,a,b,num);
    /*@formatter:on*/
    return SP_SUCCESS;

};
__global__
void spTransformAddKernel(size_type *v, size_type const *a, size_type const *b, size_type num)
{
    uint s = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (s < num) { v[s] = a[s] + b[s]; }
};

int spTransformAdd(size_type *v, size_type const *a, size_type const *b, size_type num)
{
    /*@formatter:off*/
   spTransformAddKernel<<<num/256+1,256>>>(v,a,b,num);
    /*@formatter:on*/
    return SP_SUCCESS;

}