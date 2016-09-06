//
// Created by salmon on 16-9-6.
//

extern "C"
{
#include "../../spParticle.h"
#include "../../spMesh.h"
#include "../../spParallel.h"

}

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include "../../../../../../../usr/local/cuda/include/curand_mtgp32_kernel.h"
#include "../../../../../../../usr/local/cuda/include/host_defines.h"
#include "../../../../../../../usr/local/cuda/include/device_launch_parameters.h"

__global__ void spParticleSortKernel(uint *cellStart,        // output: cell start index
                                     uint *cellEnd,          // output: cell end index
                                     uint *trashStart,
                                     uint *gridParticleHash, // input: sorted grid hashes
                                     uint *gridParticleIndex,// input: sorted particle indices
                                     uint numParticles)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements

    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint number_thread = __umul24(gridDim.x, blockDim.x);
    uint hash;

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
int spParticleSort(spParticle *sp)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    uint iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    size_type num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);

    size_type numParticles = spParticleGetNumOfParticle(sp);

    uint *hash = (uint *) spParticleGetAttributeData(sp, 0);

    uint *start_pos, *end_pos, *index;

    spParticleGetIndexArray(sp, &start_pos, &end_pos, &index);

    thrust::sort_by_key(thrust::device_ptr<uint>(hash),
                        thrust::device_ptr<uint>(hash + numParticles),
                        thrust::device_ptr<uint>(index));

    uint trashStart = 0;
    int numThreads = 256;
    uint smemSize = sizeof(uint) * (numThreads + 1);
    /*@formatter:off*/
    spParticleSortKernel<<< numParticles/numThreads+1, numThreads,smemSize >>> (start_pos, end_pos, &trashStart, hash, index, numParticles);

    /*@formatter:on*/
    SP_CALL(spParticleRemoveNull(sp, trashStart));
    return SP_SUCCESS;
};

int spParticleAutoReorder(spParticle *sp)
{
    return SP_DO_NOTHING;
};


__global__ void
spParticleSetKernel(void *dest,
                    void const *src,
                    uint head,
                    uint tail,
                    uint const *pIndex,
                    uint size_in_byte)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements

    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint number_thread = __umul24(gridDim.x, blockDim.x);
    uint hash;

    if (index < tail - head)     // handle case when no. of particles not multiple of block size
    {
        memcpy((char *) dest + __umul24(index + head, size_in_byte),
               (char *) src + __umul24(index, size_in_byte), size_in_byte);
    }

}
int spParticleSet(spParticle *sp, size_type head, size_type tail, void **data)
{
    int numThreads = 256;
    for (int i = 0; i < spParticleGetNumberOfAttributes(sp); ++i)
    {
        /*@formatter:off*/
        spParticleSetKernel<<< (tail-head)/numThreads+1, numThreads >>> (
                spParticleGetAttributeData(sp,i),data[i],
                head,tail,spParticleGetIndex(sp),
                spParticleGetAttributeTypeSizeInByte(sp,i)
        );
        /*@formatter:on*/

    }
    return SP_SUCCESS;

};
int spParticleGet(spParticle *sp, size_type head, size_type tail, void **data)
{

}


__global__ void
spParticleMemcpyKernel(void *dest,
                       void const *src,
                       const uint *gridParticleIndex,// input: sorted particle indices
                       uint numParticles, uint ele_size_in_byte)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;


    if (index < numParticles)
    {
        memcpy(dest + index * ele_size_in_byte, src + gridParticleIndex[index] * ele_size_in_byte, ele_size_in_byte);
    }
}


__global__ void
spParticleMemcpyUIntKernel(uint *dest,
                           uint const *src,
                           const uint *gridParticleIndex,// input: sorted particle indices
                           uint numParticles)
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

        if (ele_size_in_byte == sizeof(uint))
        {
            /*@formatter:off*/
            spParticleMemcpyUIntKernel<<< num_particle/numThreads+1, numThreads >>> (
                (uint *)   t_data, (uint const *) src,
                    spParticleGetIndex(sp),num_particle);
            /*@formatter:on*/


        }
        else
        {
            /*@formatter:off*/
            spParticleMemcpyKernel<<< num_particle/numThreads+1, numThreads >>> (
                t_data,  src,spParticleGetIndex(sp),num_particle,ele_size_in_byte);
            /*@formatter:on*/
        }

        SP_CALL(spParticleSetAttributeData(sp, i, t_data));

        t_data = src;
    }


    SP_CALL(spParallelDeviceFree(&t_data));
    return SP_SUCCESS;

};
int spParticleGetParticleInCell(spParticle *sp, uint num, uint *cell_hash, uint *start, uint *end, uint **index)
{
    return SP_SUCCESS;
}