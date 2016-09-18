//
// Created by salmon on 16-9-6.
//

extern "C"
{
#include <assert.h>
#include "../../spParallel.h"
#include "../../spMesh.h"
#include "../../spField.h"
#include "../../spAlogorithm.h"
#include "../../spParticle.h"

#include "../spParticle.impl.h"
#include "../sp_device.h"

}

#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
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
    int error_code = SP_SUCCESS;

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    int iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    size_type num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);

    size_type num_of_pic = spParticleGetPIC(sp);

    size_type *start_pos, *count, *sorted_id;

    SP_CALL(spParticleGetBucket(sp, &start_pos, &count, &sorted_id));

    SP_CALL(spFillSeqInt(sorted_id, spParticleGetCapacity(sp), 0, 1));

    size_type m_start[3], m_end[3], m_count[3], m_strides[3], m_dims[3];

    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, m_start, m_end, m_count));
    SP_CALL(spMeshGetStrides(m, m_strides));
    SP_CALL(spMeshGetDims(m, m_dims));

    size_type block_dim[3], grid_dim[3];
    spParallelThreadBlockDecompose(256, 3, m_start, m_end, grid_dim, block_dim);

    /*@formatter:off*/
     spParticleInitializeBucket_device_kernel<<<sizeType2Dim3(grid_dim),sizeType2Dim3(block_dim)>>>(
                        sizeType2Dim3(m_start),sizeType2Dim3(m_count),sizeType2Dim3(m_strides),num_of_pic,start_pos,count);
    /*@formatter:on*/
    return error_code;
}

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
    }
}

int spParticleBuildBucket_device(spParticle *sp)
{
    int error_code = SP_SUCCESS;

    size_type num_of_particle = spParticleGetSize(sp);

    size_type trashStart = 0;
    int numThreads = 256;
    uint smemSize = sizeof(uint) * (numThreads + 1);

    size_type *bucket_start, *bucket_end, *index;

    size_type *hash = (size_type *) spParticleGetAttributeData(sp, 0);

//    /*@formatter:off*/
//    spParticleRebuildBucketKernel<<<num_of_particle / numThreads + 1, numThreads,smemSize>>>(
//        bucket_start, bucket_count, &trashStart, hash, index, num_of_particle);
//    /*@formatter:on*/

    return error_code;
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


int spParticleCoordinateLocalToGlobal(spParticle *sp)
{
    int error_code = SP_SUCCESS;
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

    spParticleGetBucket(sp, &start_pos, &end_pos, &index);

    uint3 blockDim;
    blockDim.x = SP_NUM_OF_THREADS_PER_BLOCK;
    blockDim.y = 1;
    blockDim.z = 1;


    SP_DEVICE_CALL_KERNEL(spParticleCooridinateConvert, sizeType2Dim3(dims), blockDim,
                          (particle_head *) (p_data), real2Real3(dx), real2Real3(xmin),
                          start_pos, end_pos, index);

    return error_code;

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
                                  curandStateScrambledSobol64 *state)
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
spRandomDistributionUniformKernel(curandStateScrambledSobol64 *state, Real *data, size_type num)
{

    unsigned int total_thread_id = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
    unsigned int total_thread_num = __umul24(blockDim.x, gridDim.x);

    curandStateScrambledSobol64 local_state = state[total_thread_id];
    for (size_type i = total_thread_id; i < num; i += total_thread_num) { data[i] = curand_uniform(&local_state); }

    state[total_thread_id] = local_state;
}
__global__ void
spRandomDistributionNormalKernel(curandStateScrambledSobol64 *state, Real *data, size_type num)
{

    unsigned int total_thread_id = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
    unsigned int total_thread_num = __umul24(blockDim.x, gridDim.x);

    curandStateScrambledSobol64 local_state = state[total_thread_id];

    for (size_type i = total_thread_id; i < num; i += total_thread_num) { data[i] = curand_normal(&local_state); }

    state[total_thread_id] = local_state;
}

int spRandomMultiDistribution(Real **data, int n_dims, int const *dist_types, size_type num, size_type offset)
{
    int error_code = SP_SUCCESS;

    curandStateScrambledSobol64 *devSobol64States;

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

    {
        /* @formatter:off */
        /* Initialize the states */
         spRandomGeneratorSobolSetupKernel<<<n_threads/VECTOR_SIZE, VECTOR_SIZE>>>(
                 devDirectionVectors64,devScrambleConstants64,
                 n_dims ,offset,devSobol64States
        );
       /* @formatter:on */
    }

    for (int n = 0; n < n_dims; ++n)
    {
        switch (dist_types[n])
        {
            case SP_RAND_NORMAL:
                /* @formatter:off */
             spRandomDistributionNormalKernel<<<n_threads/VECTOR_SIZE, VECTOR_SIZE>>>(
                                          devSobol64States + n *  n_threads,data[n],num);
                /* @formatter:on */

                break;
            case SP_RAND_UNIFORM:
            default:
                /* @formatter:off */
             spRandomDistributionUniformKernel<<<n_threads/VECTOR_SIZE, VECTOR_SIZE>>>(
                                           devSobol64States + n *  n_threads,data[n],num);
                /* @formatter:on */
                break;
        }

    }


    SP_DEVICE_CALL(cudaFree((void *) (devSobol64States)));
    SP_DEVICE_CALL(cudaFree(devDirectionVectors64));
    SP_DEVICE_CALL(cudaFree(devScrambleConstants64));
    return error_code;
}

