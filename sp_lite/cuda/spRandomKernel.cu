//
// Created by salmon on 16-7-28.
//
extern "C"
{
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "spParallelCUDA.h"
#include "../spRandom.h"

}

#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include </usr/local/cuda/include/curand_kernel.h>


/* Number of 64-bit vectors per dimension */
#define VECTOR_SIZE 64

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)


typedef struct spRandomGenerator_s
{
    curandStateScrambledSobol64 *devSobol64States;

    unsigned long long int *devDirectionVectors64;
    unsigned long long int *devScrambleConstants64;

    int num_of_dimensions;
    dim3 blocks;
    dim3 threads;
    size_type num_of_threads;
} spRandomGenerator;




/**
 * This kernel initializes state per thread for each of x, y, and z,vx,vy,vz
 */
__global__ void
spRandomGeneratorSobolSetupKernel(unsigned long long *sobolDirectionVectors,
                                  unsigned long long *sobolScrambleConstants,
                                  int num_of_dim, size_type offset,
                                  curandStateScrambledSobol64 *state)
{
    size_type id =
            (threadIdx.x + blockIdx.x * blockDim.x) +
            (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x +
            (threadIdx.z + blockIdx.z * blockDim.z) * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    /* Each thread uses 3 different dimensions */
    for (int i = 0; i < num_of_dim; ++i)
    {
        curand_init(sobolDirectionVectors + VECTOR_SIZE * (id * num_of_dim + i),
                    sobolScrambleConstants[id * num_of_dim + i],
                    offset,
                    &(state[id]));
    }

}

int spRandomGeneratorCreate(spRandomGenerator **gen, int type, int num_of_dimension, size_type offset)
{

    SP_CALL(spParallelHostAlloc((void **) gen, sizeof(spRandomGenerator)));
    {
        size_type blocks[3] = {64, 1, 1};
        size_type threads[3] = {64, 1, 1};
        spRandomGeneratorSetThreadBlocks(*gen, blocks, threads);
        spRandomGeneratorSetNumOfDimensions(*gen, num_of_dimension);
    }

    curandDirectionVectors64_t *hostVectors64;
    unsigned long long int *hostScrambleConstants64;

    /* Get pointers to the 64 bit scrambled direction vectors and constants*/
    CURAND_CALL(curandGetDirectionVectors64(&hostVectors64,
                                            CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));

    CURAND_CALL(curandGetScrambleConstants64(&hostScrambleConstants64));


    /* Allocate memory for 3 states per thread (x, y, z), each state to get a unique dimension */
    CUDA_CALL(cudaMalloc((void **) &((*gen)->devSobol64States),
                         spRandomGeneratorGetNumOfThreads(*gen) *
                         spRandomGeneratorGetNumOfDimensions(*gen) *
                         sizeof(curandStateScrambledSobol64)));

    /* Allocate memory and copy 3 sets of vectors per thread to the device */

    CUDA_CALL(cudaMalloc((void **) &((*gen)->devDirectionVectors64),
                         spRandomGeneratorGetNumOfThreads(*gen) *
                         spRandomGeneratorGetNumOfDimensions(*gen) * VECTOR_SIZE * sizeof(long long int)));

    CUDA_CALL(cudaMemcpy((*gen)->devDirectionVectors64, hostVectors64,
                         spRandomGeneratorGetNumOfThreads(*gen) *
                         spRandomGeneratorGetNumOfDimensions(*gen) * VECTOR_SIZE * sizeof(long long int),
                         cudaMemcpyHostToDevice));

    /* Allocate memory and copy 6 scramble constants (one costant per dimension)
       per thread to the device */

    CUDA_CALL(cudaMalloc((void **) &((*gen)->devScrambleConstants64),
                         spRandomGeneratorGetNumOfThreads(*gen) *
                         spRandomGeneratorGetNumOfDimensions(*gen) * sizeof(long long int)));

    CUDA_CALL(cudaMemcpy((*gen)->devScrambleConstants64, hostScrambleConstants64,
                         spRandomGeneratorGetNumOfThreads(*gen) *
                         spRandomGeneratorGetNumOfDimensions(*gen) * sizeof(long long int),
                         cudaMemcpyHostToDevice));

    {
        size_type s_blocks[3], s_threads[3];
        spRandomGeneratorGetThreadBlocks(*gen, s_blocks, s_threads);
        dim3 blocks = sizeType2Dim3(s_blocks), threads = sizeType2Dim3(s_threads);
        /* @formatter:off */
        /* Initialize the states */
         spRandomGeneratorSobolSetupKernel<<<blocks, threads>>>((*gen)->devDirectionVectors64,
       (*gen)-> devScrambleConstants64,
        spRandomGeneratorGetNumOfDimensions(*gen) ,offset,
       (*gen)-> devSobol64States
        );
       /* @formatter:on */
    }
    return SP_SUCCESS;
}

int spRandomGeneratorDestroy(spRandomGenerator **gen)
{
    CUDA_CALL(cudaFree((*gen)->devSobol64States));
    CUDA_CALL(cudaFree((*gen)->devDirectionVectors64));
    CUDA_CALL(cudaFree((*gen)->devScrambleConstants64));
    return spParallelHostFree((void **) gen);
}

int spRandomGeneratorSetNumOfDimensions(spRandomGenerator *gen, int n)
{
    gen->num_of_dimensions = n;
    return SP_SUCCESS;
}

int spRandomGeneratorGetNumOfDimensions(spRandomGenerator const *gen)
{
    return gen->num_of_dimensions;
}

int spRandomGeneratorSetThreadBlocks(spRandomGenerator *gen, size_type const *blocks, size_type const *threads)
{
    gen->blocks = sizeType2Dim3(blocks);
    gen->threads = sizeType2Dim3(threads);

    gen->num_of_threads = blocks[0] * blocks[1] * blocks[2] * threads[0] * threads[1] * threads[2];
}

int spRandomGeneratorGetThreadBlocks(spRandomGenerator *gen, size_type *blocks, size_type *threads)
{
    blocks[0] = gen->blocks.x;
    blocks[1] = gen->blocks.y;
    blocks[2] = gen->blocks.z;
    threads[0] = gen->threads.x;
    threads[1] = gen->threads.y;
    threads[2] = gen->threads.z;

    return SP_SUCCESS;
}

size_type spRandomGeneratorGetNumOfThreads(spRandomGenerator const *gen)
{
    return gen->num_of_threads;
}

__global__ void
spRandomDistributionInCellUniformKernel(curandStateScrambledSobol64 *state, Real *data, dim3 min, dim3 max,
                                        dim3 strides,
                                        size_type num)
{

    size_type total_thread_id =
            (threadIdx.x + blockIdx.x * blockDim.x) +
            (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x +
            (threadIdx.z + blockIdx.z * blockDim.z) * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    curandStateScrambledSobol64 local_state = state[total_thread_id];

    size_type num_of_thread = blockDim.z * blockDim.x * blockDim.y;

    for (int x = blockIdx.x + min.x; x < max.x; ++x)
        for (int y = blockIdx.y + min.y; y < max.y; ++y)
            for (int z = blockIdx.z + min.z; z < max.z; ++z)
            {
                Real *local_data = data +
                                   (blockIdx.x + min.x) * strides.x +
                                   (blockIdx.y + min.y) * strides.y +
                                   (blockIdx.z + min.z) * strides.z;
                /* Generate quasi-random double precision coordinates */
                for (size_type s = 0; s < num; s += num_of_thread)
                {
                    local_data[s] = curand_uniform(&local_state);
                }
            }

    state[total_thread_id] = local_state;
}


__global__ void
spRandomDistributionInCellNormalKernel(curandStateScrambledSobol64 *state, Real *data, dim3 min, dim3 max, dim3 strides,
                                       size_type num)
{

    size_type total_thread_id =
            (threadIdx.x + blockIdx.x * blockDim.x) +
            (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x +
            (threadIdx.z + blockIdx.z * blockDim.z) * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    curandStateScrambledSobol64 local_state = state[total_thread_id];

    size_type num_of_thread = blockDim.z * blockDim.x * blockDim.y;

    for (int x = blockIdx.x + min.x; x < max.x; ++x)
        for (int y = blockIdx.y + min.y; y < max.y; ++y)
            for (int z = blockIdx.z + min.z; z < max.z; ++z)
            {
                Real *local_data = data +
                                   (blockIdx.x + min.x) * strides.x +
                                   (blockIdx.y + min.y) * strides.y +
                                   (blockIdx.z + min.z) * strides.z;
                /* Generate quasi-random double precision coordinates */
                for (size_type s = 0; s < num; s += num_of_thread)
                {
                    local_data[s] = curand_normal(&local_state);
                }
            }

    state[total_thread_id] = local_state;
}


/**
 * data[i][s]=a*dist(rand())+b;
 * @param gen
 * @param data
 * @param num_of_dimension
 * @param num_of_sample
 * @param u0
 * @param sigma
 * @return
 */
int
spRandomDistributionInCell(spRandomGenerator *gen, int const *dist_types, Real **data,
                           size_type const *min, size_type const *max, size_type const *strides,
                           size_type num_per_cell)
{
    size_type s_blocks[3], s_threads[3];
    spRandomGeneratorGetThreadBlocks(gen, s_blocks, s_threads);

    dim3 blocks = sizeType2Dim3(s_blocks), threads = sizeType2Dim3(s_threads);
    for (int n = 0; n < spRandomGeneratorGetNumOfDimensions(gen); ++n)
    {
        switch (dist_types[n])
        {
            case SP_RAND_NORMAL:
                /* @formatter:off */
                spRandomDistributionInCellNormalKernel<<<blocks, threads>>>(
                        gen->devSobol64States+n*spRandomGeneratorGetNumOfThreads(gen),
                        data[n],sizeType2Dim3(min),sizeType2Dim3(max),sizeType2Dim3(strides),num_per_cell);
                /* @formatter:on */
                break;
            case SP_RAND_UNIFORM:
            default:
                /* @formatter:off */
             spRandomDistributionInCellUniformKernel<<<blocks, threads>>>(
                        gen->devSobol64States+n*spRandomGeneratorGetNumOfThreads(gen),
                        data[n] ,sizeType2Dim3(min),sizeType2Dim3(max),sizeType2Dim3(strides),num_per_cell);

                /* @formatter:off */
                break;
        }
    }

    return SP_SUCCESS;
}
