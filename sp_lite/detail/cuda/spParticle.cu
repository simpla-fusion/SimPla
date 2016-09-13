//
// Created by salmon on 16-9-6.
//

extern "C"
{
#include "../../spParallel.h"
#include "../../spMesh.h"
#include "../../spParticle.h"
#include "../../spParticle.impl.h"
#include "../../spField.h"
#include "../sp_device.h"

}


#include </usr/local/cuda/include/host_defines.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include "../../spDataType.h"
#include "../../spAlogorithm.h"

__global__ void spParticleRebuildBucketKernel(size_type *cellStart,        // output: cell start index
                                              size_type *cellEnd,          // output: cell end index
                                              size_type *trashStart,
                                              size_type *gridParticleHash, // input: sorted grid hashes
                                              size_type *gridParticleIndex,// input: sorted particle indices
                                              size_type numParticles)
{
    extern __shared__ size_type sharedHash[];    // blockSize + 1 elements

    size_type index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    size_type number_thread = __umul24(gridDim.x, blockDim.x);
    size_type hash;

    if (index < numParticles)     // handle case when no. of particles not multiple of block size
    {
        hash = gridParticleHash[gridParticleIndex[index]];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[gridParticleIndex[index - 1]];
        }
    }

    __syncthreads();

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            if (hash != -1) { cellStart[hash] = index; } else { *trashStart = index; }
            if (index > 0) cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1) { if (hash != -1) { cellEnd[hash] = index + 1; }}

//        // Now use the sorted index to reorder the pos and vel data
//        uint sortedIndex = gridParticleIndex[index];
//        float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
//        float4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh
//
//        sortedPos[index] = pos;
//        sortedVel[index] = vel;
    }
}
int spParticleResetHash(spParticle *sp)
{
    UNIMPLEMENTED;
    return SP_DO_NOTHING;
}

int spParticleBuildBucketFromIndex(spParticle *sp)
{

    size_type num_of_particle = spParticleGetSize(sp);

    size_type trashStart = 0;
    int numThreads = 256;
    uint smemSize = sizeof(uint) * (numThreads + 1);

    size_type *bucket_start, *bucket_end, *index;

    spParticleGetBucketIndex(sp, &bucket_start, &bucket_end, &index);

    size_type *hash = (size_type *) spParticleGetAttributeData(sp, 0);

    /*@formatter:off*/
    spParticleRebuildBucketKernel<<<num_of_particle / numThreads + 1, numThreads,smemSize>>>(
        bucket_start, bucket_end, &trashStart, hash, index, num_of_particle);
    /*@formatter:on*/
}

SP_DEVICE_DECLARE_KERNEL (spParticleCooridinateConvert,
                          particle_head *sp,
                          Real3 dx, Real3 min,
                          size_type const *start_pos,
                          size_type const *end_pos,
                          size_type const *sorted_index
)
{

    uint s0 = __umul24(blockIdx.x, gridDim.x) +
        __umul24(blockIdx.y, gridDim.y) +
        __umul24(blockIdx.z, gridDim.z);

    __shared__ Real x0, y0, z0;

    if (threadIdx.x == 0)
    {
        x0 = blockIdx.x * dx.x + min.x;
        y0 = blockIdx.y * dx.y + min.y;
        z0 = blockIdx.z * dx.z + min.z;
    }

        spParallelSyncThreads();

    if (start_pos[s0] + threadIdx.x < end_pos[s0])
    {
        size_type s = sorted_index[start_pos[s0] + threadIdx.x];
        sp->rx[s] += x0;
        sp->ry[s] += y0;
        sp->rz[s] += z0;
    }
};


int spParticleCooridinateLocalToGlobal(spParticle *sp)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    uint iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    Real dx[3], xmin[3], xmax[3];
    size_type dims[3];

    spMeshGetDims(m, dims);
    spMeshGetDx(m, dx);
    spMeshGetBox(m, SP_DOMAIN_ALL, xmin, xmax);

    void **p_data;

    SP_CALL(spParticleGetAllAttributeData_device(sp, &p_data));

    size_type *start_pos, *end_pos, *index;

    spParticleGetBucketIndex(sp, &start_pos, &end_pos, &index);

    uint3 blockDim;
    blockDim.x = SP_NUM_OF_THREADS_PER_BLOCK;
    blockDim.y = 1;
    blockDim.z = 1;


    SP_DEVICE_CALL_KERNEL(spParticleCooridinateConvert, sizeType2Dim3(dims), blockDim,
                          (particle_head *) (p_data), real2Real3(dx), real2Real3(xmin),
                          start_pos, end_pos, index);

    return SP_SUCCESS;

};


__global__ void
spParticleMemcpyKernel(void *dest,
                       void const *src,
                       const uint *gridParticleIndex,// input: sorted particle indices
                       uint numParticles, uint ele_size_in_byte)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;


    if (index < numParticles)
    {
        memcpy(dest + index * ele_size_in_byte,
               src + gridParticleIndex[index] * ele_size_in_byte, ele_size_in_byte);
    }
}


__global__ void
spParticleMemcpyUIntKernel(uint *dest,
                           uint const *src,
                           const uint *gridParticleIndex,// input: sorted particle indices
                           size_type numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index < numParticles) { dest[index] = src[gridParticleIndex[index]]; }
}

int spParticleReorder(spParticle *sp)
{
    int numThreads = 256;
    size_type max_num_particle = spParticleGetMaxNumOfParticle(sp);
    size_type num_particle = spParticleGetNumOfParticle(sp);

    size_type size_in_byte = 0;
    void *t_data = NULL;
    for (int i = 0; i < spParticleGetNumberOfAttributes(sp); ++i)
    {
        size_type ele_size_in_byte = spParticleGetAttributeTypeSizeInByte(sp, i);

        size_type t_size = size_in_byte * ele_size_in_byte;

        if (size_in_byte != t_size)
        {
            size_in_byte = t_size;
            if (t_data != NULL) { spParallelDeviceFree(&t_data); }
            spParallelDeviceAlloc(&t_data, size_in_byte);
        }

        void *src = spParticleGetAttributeData(sp, i);

//        if (ele_size_in_byte == sizeof(uint))
//        {
//            SP_DEVICE_CALL_KERNEL(spParticleMemcpyUIntKernel, num_particle / numThreads + 1, numThreads,
//                                  (size_type *) t_data, (uint const *) src, spParticleGetSortedIndex(sp),
//                                  num_particle);
//
//        }
//        else
//        {
//            SP_DEVICE_CALL_KERNEL(spParticleMemcpyKernel, num_particle / numThreads + 1, numThreads,
//                                  t_data, src, spParticleGetSortedIndex(sp), num_particle, ele_size_in_byte);
//        }

        SP_CALL(spParticleSetAttributeData(sp, i, t_data));

        t_data = src;
    }


    SP_CALL(spParallelDeviceFree(&t_data));
    return SP_SUCCESS;

};

int spParticleGetCell(spParticle *sp, uint num, uint *cell_hash, uint *start, uint *end, uint **index)
{
    return SP_SUCCESS;
}