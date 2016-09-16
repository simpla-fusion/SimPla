//
// Created by salmon on 16-9-13.
//
extern "C"
{
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
void spMemoryRelativeCopyKernel(Real *dest, Real const *src, size_type num, size_type max, size_type const *index)
{
    uint s = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (s < num)
    {
        assert(index[s] < max);
        dest[s] = src[index[s]];
    }
};

int spMemoryRelativeCopy(Real *dest, Real const *src, size_type num, size_type max, size_type const *index)
{
    /*@formatter:off*/
   spMemoryRelativeCopyKernel<<<num/256+1,256>>>(dest,src,num,max,index);
    /*@formatter:on*/
    return SP_SUCCESS;
}

__global__
void spFillSeqIntKernel(size_type *v, size_type num, size_type min)
{
    uint s = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (s < num) { v[s] = s + min; }
};

int spFillSeqInt(size_type *v, size_type num, size_type min)
{
    /*@formatter:off*/
   spFillSeqIntKernel<<<num/256+1,256>>>(v,num,min);
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