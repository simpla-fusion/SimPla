//
// Created by salmon on 16-9-6.
//

extern "C"
{
#include <assert.h>

#include "../../spParallel.h"
#include "../../spMesh.h"
#include "../../spField.h"
#include "../../spParticle.h"
#include "../../spAlogorithm.h"
#include "../spParticle.impl.h"
#include "../sp_device.h"

}


#include </usr/local/cuda/include/curand_kernel.h>

__global__ void
spParticleInitializeBucket_device_kernel(dim3 start,
                                         dim3 count,
                                         dim3 strides,
                                         int num_pic,
                                         size_type *start_pos,
                                         size_type *f_count)
{

    uint x = __umul24(blockIdx.x, blockIdx.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockIdx.y) + threadIdx.y;
    uint z = __umul24(blockIdx.z, blockIdx.z) + threadIdx.z;

    if (x < count.x && y < count.y && z < count.z)
    {
        uint s = __umul24(start.x + x, strides.x) +
                 __umul24(start.y + y, strides.y) +
                 __umul24(start.z + z, strides.z);
        start_pos[s] = (x * count.y * count.z + y * count.z + z) * num_pic;
        f_count[s] = (size_type) num_pic;
    }

}

int spParticleInitializeBucket_device(spParticle *sp)
{

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    int iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    size_type num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);

    size_type num_of_pic = spParticleGetPIC(sp);

    size_type *bucket_start, *bucket_count, *sorted_id, *cell_hash;

    SP_CALL(spParticleGetBucket(sp, &bucket_start, &bucket_count, &sorted_id, &cell_hash));

    SP_CALL(spFillSeqInt(sorted_id, spParticleCapacity(sp), 0, 1));
    SP_CALL(spMemSet(cell_hash, -1, spParticleCapacity(sp) * sizeof(size_type)));
    size_type m_start[3], m_end[3], m_count[3], m_strides[3], m_dims[3];

    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, m_start, m_end, m_count));
    SP_CALL(spMeshGetStrides(m, m_strides));
    SP_CALL(spMeshGetDims(m, m_dims));

    size_type block_dim[3], grid_dim[3];
    SP_CALL(spParallelThreadBlockDecompose(256, 3, m_start, m_end, grid_dim, block_dim));

    SP_CALL_DEVICE_KERNEL(spParticleInitializeBucket_device_kernel,
                          sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          sizeType2Dim3(m_start), sizeType2Dim3(m_count),
                          sizeType2Dim3(m_strides), num_of_pic, bucket_start, bucket_count);

    return SP_SUCCESS;
}

__global__ void
spParticleInitializeHash_kernel(size_type *id,
                                dim3 strides,
                                size_type const *bucket_start,
                                size_type const *bucket_count,
                                size_type const *sort_id)
{


    uint s = __umul24(blockIdx.x, strides.x) +
             __umul24(blockIdx.y, strides.y) +
             __umul24(blockIdx.z, strides.z);

    if (threadIdx.x < bucket_count[s]) { id[sort_id[bucket_start[s] + threadIdx.x]] = s; }

}

int spParticleUpdateHash(spParticle *sp)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    int iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    size_type m_start[3], m_end[3], m_count[3], m_strides[3], m_dims[3];
    size_type *bucket_start, *bucket_count, *sorted_id;

    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, m_start, m_end, m_count));
    SP_CALL(spMeshGetStrides(m, m_strides));
    SP_CALL(spMeshGetDims(m, m_dims));

    size_type block_dim[3], grid_dim[3];
    SP_CALL(spParallelThreadBlockDecompose(256, 3, m_start, m_end, grid_dim, block_dim));

    assert(spParticleGetPIC(sp) < 256);


    SP_CALL_DEVICE_KERNEL(spParticleInitializeHash_kernel,
                          sizeType2Dim3(m_dims), 256,
                          (size_type *) spParticleGetAttributeData(sp, 0),
                          sizeType2Dim3(m_strides), bucket_start, bucket_count, sorted_id);

    return SP_SUCCESS;

}

/**
 *  copy from cuda example/particle
 */
__global__ void
spParticleRebuildBucket_kernel(size_type *cellStart,        // output: cell start index
                               size_type *cellEnd,          // output: cell end index
                               size_type const *gridParticleHash, // input: sorted grid hashes
                               size_type const *gridParticleIndex,// input: sorted particle indices
                               size_type numParticles, size_type num_cell)
{


    extern __shared__ size_type sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    size_type hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index - 1];
        }
    }

    spParallelSyncThreads();

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            assert(hash + 1 <= num_cell);

            cellStart[hash + 1] = index;

            if (index > 0)
            {

                assert(sharedHash[threadIdx.x] + 1 <= num_cell);

                cellEnd[sharedHash[threadIdx.x] + 1] = index;

            }
        }

        if (index == numParticles - 1)
        {
            assert(hash + 1 <= num_cell);
            cellEnd[hash + 1] = index + 1;
        }

    }
}


__global__ void
_CopyBucketStartCount_kernel(size_type *b_start,        // output: cell start index
                             size_type *b_end,          // output: cell end index
                             size_type *start,
                             size_type *count,
                             size_type num_of_cell)
{

    size_type index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index < num_of_cell)
    {
        start[index] = b_start[index + 1];
        count[index] = b_end[index + 1] - b_start[index + 1];
        assert(count[index] >= 0);
    }
}

__host__
int spParticleBuildBucket_device(spParticle *sp)
{

    assert(spParticleNeedSorting(sp) == SP_FALSE);

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    int iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    size_type num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);

    size_type num_of_particle = spParticleSize(sp);

    size_type *bucket_start, *bucket_count, *sorted_idx, *cell_hash;

    SP_CALL(spParticleGetBucket(sp, &bucket_start, &bucket_count, &sorted_idx, &cell_hash));

    size_type *b_start, *b_end;
    SP_CALL(spMemDeviceAlloc((void **) &b_end, (num_of_cell + 1) * sizeof(size_type)));
    SP_CALL(spMemDeviceAlloc((void **) &b_start, (num_of_cell + 1) * sizeof(size_type)));
    SP_CALL(spMemSet(b_start, 0, (num_of_cell + 1) * sizeof(size_type)));
    SP_CALL(spMemSet(b_end, 0, (num_of_cell + 1) * sizeof(size_type)));

    int numThreads = 256;
    uint sMemSize = sizeof(size_type) * (numThreads + 1);


    SP_DEVICE_CALL_KERNEL2(spParticleRebuildBucket_kernel, num_of_particle / numThreads + 1, numThreads, sMemSize,
                           b_start, b_end, cell_hash, sorted_idx, num_of_particle, num_of_cell);

    SP_CALL_DEVICE_KERNEL(_CopyBucketStartCount_kernel,
                          num_of_cell / numThreads + 1, num_of_cell,
                          b_start, b_end, bucket_start, bucket_count, num_of_cell);


    SP_CALL(spMemCopy(&num_of_particle, b_start, sizeof(size_type)));

    if (num_of_particle != 0) {SP_CALL(spParticleResize(sp, num_of_particle)); }

    SP_CALL(spMemDeviceFree((void **) &b_start));
    SP_CALL(spMemDeviceFree((void **) &b_end));
    return SP_SUCCESS;
}

__global__ void
spParticleCoordinateConvert(particle_head *sp,
                            Real3 dx, Real3 min,
                            size_type const *start_pos,
                            size_type const *end_pos,
                            size_type const *sorted_index)
{
    uint s0 = __umul24(blockIdx.x, gridDim.x) + __umul24(blockIdx.y, gridDim.y) + __umul24(blockIdx.z, gridDim.z);
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


int spParticleCoordinateLocalToGlobal(spParticle *sp)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    uint iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    Real dx[3], xmin[3], xmax[3];
    size_type dims[3];

    SP_CALL(spMeshGetDims(m, dims));
    SP_CALL(spMeshGetDx(m, dx));
    SP_CALL(spMeshGetBox(m, SP_DOMAIN_ALL, xmin, xmax));

    void **p_data;

    SP_CALL(spParticleGetAllAttributeData_device(sp, &p_data, NULL));

    size_type *start_pos, *end_pos, *index;

    SP_CALL(spParticleGetBucket(sp, &start_pos, &end_pos, &index, NULL));

    uint3 blockDim;
    blockDim.x = SP_NUM_OF_THREADS_PER_BLOCK;
    blockDim.y = 1;
    blockDim.z = 1;


    SP_CALL_DEVICE_KERNEL(spParticleCoordinateConvert, sizeType2Dim3(dims), blockDim,
                          (particle_head *) (p_data), real2Real3(dx), real2Real3(xmin),
                          start_pos, end_pos, index);

    return SP_SUCCESS;

};



/* Number of 64-bit vectors per dimension */
#define VECTOR_SIZE 64

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)



/**
 * This kernel initializes state per thread for each of x, y, and z,vx,vy,vz
 */
__global__ void
spRandomGeneratorSobolSetupKernel(unsigned long long *sobolDirectionVectors,
                                  unsigned long long *sobolScrambleConstants,
                                  int num_of_dim, size_type offset,
                                  struct curandStateScrambledSobol64 *state)
{
    unsigned int id = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
    /* Each thread uses 3 different dimensions */
    for (int i = 0; i < num_of_dim; ++i)
    {
        curand_init(sobolDirectionVectors + VECTOR_SIZE * (id * num_of_dim + i),
                    sobolScrambleConstants[id * num_of_dim + i],
                    offset,
                    &(state[id * num_of_dim + i]));
    }

}


__global__ void
spRandomDistributionUniformKernel(struct curandStateScrambledSobol64 *state, Real *data, size_type num)
{

    unsigned int total_thread_id = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
    unsigned int total_thread_num = __umul24(blockDim.x, gridDim.x);

    struct curandStateScrambledSobol64 local_state = state[total_thread_id];
    for (size_type i = total_thread_id; i < num; i += total_thread_num) { data[i] = curand_uniform(&local_state); }

    state[total_thread_id] = local_state;
}

__global__ void
spRandomDistributionNormalKernel(struct curandStateScrambledSobol64 *state, Real *data, size_type num)
{

    unsigned int total_thread_id = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
    unsigned int total_thread_num = __umul24(blockDim.x, gridDim.x);

    struct curandStateScrambledSobol64 local_state = state[total_thread_id];

    for (size_type i = total_thread_id; i < num; i += total_thread_num) { data[i] = curand_normal(&local_state); }

    state[total_thread_id] = local_state;
}

int spParticleInitialize_device(Real **data, int n_dims, int const *dist_types, size_type num, size_type offset)
{

    int error_code = SP_SUCCESS;

    struct curandStateScrambledSobol64 *devSobol64States;

    unsigned long long int *devDirectionVectors64;
    unsigned long long int *devScrambleConstants64;

    size_type n_threads = 64 * VECTOR_SIZE;

    curandDirectionVectors64_t *hostVectors64;
    unsigned long long int *hostScrambleConstants64;

    /* Get pointers to the 64 bit scrambled direction vectors and constants*/
    CURAND_CALL(curandGetDirectionVectors64(&hostVectors64,
                                            CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));

    CURAND_CALL(curandGetScrambleConstants64(&hostScrambleConstants64));


    /* Allocate memory for 3 states per thread (x, y, z), each state to get a unique dimension */
    SP_DEVICE_CALL(cudaMalloc((void **) &(devSobol64States), n_threads * n_dims * sizeof(curandStateScrambledSobol64)));

    /* Allocate memory and copy 3 sets of vectors per thread to the detail */

    SP_DEVICE_CALL(cudaMalloc((void **) &(devDirectionVectors64),
                              n_threads * n_dims * VECTOR_SIZE * sizeof(long long int)));

    SP_DEVICE_CALL(cudaMemcpy(devDirectionVectors64, hostVectors64,
                              n_threads * n_dims * VECTOR_SIZE * sizeof(long long int),
                              cudaMemcpyHostToDevice));

    /* Allocate memory and copy 6 scramble constants (one costant per dimension)
       per thread to the detail */

    SP_DEVICE_CALL(cudaMalloc((void **) &(devScrambleConstants64),
                              n_threads * n_dims * sizeof(long long int)));

    SP_DEVICE_CALL(cudaMemcpy(devScrambleConstants64, hostScrambleConstants64,
                              n_threads * n_dims * sizeof(long long int),
                              cudaMemcpyHostToDevice));


    /* Initialize the states */
    SP_CALL_DEVICE_KERNEL(spRandomGeneratorSobolSetupKernel,
                          n_threads / VECTOR_SIZE, VECTOR_SIZE,
                          devDirectionVectors64, devScrambleConstants64,
                          n_dims, offset, devSobol64States);


    for (int n = 0; n < n_dims; ++n)
    {
        switch (dist_types[n])
        {
            case SP_RAND_NORMAL: //

            SP_CALL_DEVICE_KERNEL(spRandomDistributionNormalKernel,
                                  (n_threads / VECTOR_SIZE), VECTOR_SIZE,
                                  (devSobol64States + n * n_threads), data[n], num);

                break;
            case SP_RAND_UNIFORM:
            default://
            SP_CALL_DEVICE_KERNEL(spRandomDistributionUniformKernel,
                                  (n_threads / VECTOR_SIZE), VECTOR_SIZE,
                                  (devSobol64States + n * n_threads), data[n], num);
                break;
        }

    }


    SP_DEVICE_CALL(cudaFree((void *) (devSobol64States)));
    SP_DEVICE_CALL(cudaFree(devDirectionVectors64));
    SP_DEVICE_CALL(cudaFree(devScrambleConstants64));
    return error_code;
}

