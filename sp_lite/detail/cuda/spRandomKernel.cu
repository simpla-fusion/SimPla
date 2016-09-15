//
// Created by salmon on 16-7-28.
//
extern "C"
{
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "spParallelCUDA.h"
#include "../../spRandom.h"
#include "../sp_device.h"

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
    size_type blocks[3];
    size_type threads[3];
    size_type num_of_threads;
} spRandomGenerator;


int spRandomGeneratorSetThreadBlocks(spRandomGenerator *gen, size_type const *blocks, size_type const *threads)
{
    gen->blocks[0] = blocks[0];
    gen->blocks[1] = blocks[1];
    gen->blocks[2] = blocks[2];
    gen->threads[0] = threads[0];
    gen->threads[1] = threads[1];
    gen->threads[2] = threads[2];
    gen->num_of_threads = blocks[0] * blocks[1] * blocks[2] * threads[0] * threads[1] * threads[2];
    return SP_SUCCESS;
}

int spRandomGeneratorGetThreadBlocks(spRandomGenerator *gen, size_type *blocks, size_type *threads)
{
    blocks[0] = gen->blocks[0];
    blocks[1] = gen->blocks[1];
    blocks[2] = gen->blocks[2];
    threads[0] = gen->threads[0];
    threads[1] = gen->threads[1];
    threads[2] = gen->threads[2];
    return SP_SUCCESS;
}


/**
 * This kernel initializes state per thread for each of x, y, and z,vx,vy,vz
 */
SP_DEVICE_GLOBAL void
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
                    &(state[id * num_of_dim + i]));
    }

}

int spRandomGeneratorCreate(spRandomGenerator **gen, int type, int num_of_dimension, size_type offset)
{
    int error_code = SP_SUCCESS;
    SP_CALL(spParallelHostAlloc((void **) gen, sizeof(spRandomGenerator)));
    {
        size_type blocks[3] = {16, 1, 1};
        size_type threads[3] = {64, 1, 1};
        spRandomGeneratorSetThreadBlocks(*gen, blocks, threads);
        spRandomGeneratorSetNumOfDimensions(*gen, num_of_dimension);
    }
    int n_dims = spRandomGeneratorGetNumOfDimensions(*gen);
    size_type n_threads = spRandomGeneratorGetNumOfThreads(*gen);
    curandDirectionVectors64_t *hostVectors64;
    unsigned long long int *hostScrambleConstants64;

    /* Get pointers to the 64 bit scrambled direction vectors and constants*/
    CURAND_CALL(curandGetDirectionVectors64(&hostVectors64,
                                            CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));

    CURAND_CALL(curandGetScrambleConstants64(&hostScrambleConstants64));


    /* Allocate memory for 3 states per thread (x, y, z), each state to get a unique dimension */
    SP_DEVICE_CALL(cudaMalloc((void **) &((*gen)->devSobol64States),
                              n_threads * n_dims *
                              sizeof(curandStateScrambledSobol64)));

    /* Allocate memory and copy 3 sets of vectors per thread to the detail */

    SP_DEVICE_CALL(cudaMalloc((void **) &((*gen)->devDirectionVectors64),
                              n_threads * n_dims * VECTOR_SIZE * sizeof(long long int)));

    SP_DEVICE_CALL(cudaMemcpy((*gen)->devDirectionVectors64, hostVectors64,
                              n_threads * n_dims * VECTOR_SIZE * sizeof(long long int),
                              cudaMemcpyHostToDevice));

    /* Allocate memory and copy 6 scramble constants (one costant per dimension)
       per thread to the detail */

    SP_DEVICE_CALL(cudaMalloc((void **) &((*gen)->devScrambleConstants64),
                              n_threads * n_dims * sizeof(long long int)));

    SP_DEVICE_CALL(cudaMemcpy((*gen)->devScrambleConstants64, hostScrambleConstants64,
                              n_threads * n_dims * sizeof(long long int),
                              cudaMemcpyHostToDevice));

    {
        size_type s_blocks[3], s_threads[3];
        spRandomGeneratorGetThreadBlocks(*gen, s_blocks, s_threads);
        /* @formatter:off */
        /* Initialize the states */
         spRandomGeneratorSobolSetupKernel<<<sizeType2Dim3(s_blocks), sizeType2Dim3(s_threads)>>>(
                 (*gen)->devDirectionVectors64,
                 (*gen)-> devScrambleConstants64,
                 spRandomGeneratorGetNumOfDimensions(*gen) ,offset,
                 (*gen)-> devSobol64States
        );
       /* @formatter:on */
    }
    return error_code;
}

int spRandomGeneratorDestroy(spRandomGenerator **gen)
{
    int error_code = SP_SUCCESS;

    if (gen != NULL && *gen != NULL && (*gen)->devSobol64States != NULL)
    {
        SP_DEVICE_CALL(cudaFree((void *) ((*gen)->devSobol64States)));
        SP_DEVICE_CALL(cudaFree((*gen)->devDirectionVectors64));
        SP_DEVICE_CALL(cudaFree((*gen)->devScrambleConstants64));
    }
    SP_CALL(spParallelHostFree((void **) gen));
    return error_code;
}

int spRandomGeneratorSetNumOfDimensions(spRandomGenerator *gen, int n)
{
    gen->num_of_dimensions = n;
    return SP_SUCCESS;
}

int spRandomGeneratorGetNumOfDimensions(spRandomGenerator const *gen) { return gen->num_of_dimensions; }


size_type spRandomGeneratorGetNumOfThreads(spRandomGenerator const *gen) { return gen->num_of_threads; }

SP_DEVICE_GLOBAL void
spRandomDistributionInCellUniformKernel(curandStateScrambledSobol64 *state, Real *data, dim3 min, dim3 max,
                                        dim3 strides, size_type num)
{

    size_type total_thread_id =
            (threadIdx.x + blockIdx.x * blockDim.x) +
            (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x +
            (threadIdx.z + blockIdx.z * blockDim.z) * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    size_type threadId = (threadIdx.x) +
                         (threadIdx.y) * blockDim.x +
                         (threadIdx.z) * blockDim.x * blockDim.y;

    curandStateScrambledSobol64 local_state = state[total_thread_id];

    size_type num_of_thread = blockDim.z * blockDim.x * blockDim.y;

    for (int x = blockIdx.x + min.x; x < max.x; x += gridDim.x)
        for (int y = blockIdx.y + min.y; y < max.y; y += gridDim.y)
            for (int z = blockIdx.z + min.z; z < max.z; z += gridDim.z)
            {
                size_type s0 = threadId + x * strides.x + y * strides.y + z * strides.z;

                /* Generate quasi-random double precision coordinates */
                for (size_type s = 0; s < num; s += num_of_thread) { data[s0 + s] = curand_uniform(&local_state); }
            }

    state[total_thread_id] = local_state;
}


SP_DEVICE_GLOBAL void
spRandomDistributionInCellNormalKernel(curandStateScrambledSobol64 *state, Real *data, dim3 min, dim3 max, dim3 strides,
                                       size_type num)
{

    size_type total_thread_id =
            (threadIdx.x + blockIdx.x * blockDim.x) +
            (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x +
            (threadIdx.z + blockIdx.z * blockDim.z) * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    size_type threadId = (threadIdx.x) +
                         (threadIdx.y) * blockDim.x +
                         (threadIdx.z) * blockDim.x * blockDim.y;

    curandStateScrambledSobol64 local_state = state[total_thread_id];

    size_type num_of_thread = blockDim.z * blockDim.x * blockDim.y;

    for (int x = blockIdx.x + min.x; x < max.x; x += gridDim.x)
        for (int y = blockIdx.y + min.y; y < max.y; y += gridDim.y)
            for (int z = blockIdx.z + min.z; z < max.z; z += gridDim.z)
            {
                size_type s0 = threadId + x * strides.x + y * strides.y + z * strides.z;

                /* Generate quasi-random double precision coordinates */
                for (size_type s = 0; s < num; s += num_of_thread) { data[s0 + s] = curand_normal(&local_state); }
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
spRandomMultiDistributionInCell(spRandomGenerator *gen, int const *dist_types, Real **data,
                                size_type const *min, size_type const *max, size_type const *strides,
                                size_type num_per_cell)
{
    int error_code = SP_SUCCESS;

    size_type s_blocks[3], s_threads[3];
    SP_CALL(spRandomGeneratorGetThreadBlocks(gen, s_blocks, s_threads));

    int n_dims = spRandomGeneratorGetNumOfDimensions(gen);
    size_type n_threads = spRandomGeneratorGetNumOfThreads(gen);
    for (int n = 0; n < n_dims; ++n)
    {
        switch (dist_types[n])
        {
            case SP_RAND_NORMAL: SP_DEVICE_CALL_KERNEL(spRandomDistributionInCellNormalKernel,
                                                       sizeType2Dim3(s_blocks), sizeType2Dim3(s_threads),
                                                       gen->devSobol64States + n * n_threads,
                                                       data[n],
                                                       sizeType2Dim3(min), sizeType2Dim3(max), sizeType2Dim3(strides), num_per_cell);
                break;
            case SP_RAND_UNIFORM:
            default: SP_DEVICE_CALL_KERNEL(spRandomDistributionInCellUniformKernel,
                                           sizeType2Dim3(s_blocks), sizeType2Dim3(s_threads),
                                           gen->devSobol64States + n * n_threads,
                                           data[n],
                                           sizeType2Dim3(min), sizeType2Dim3(max), sizeType2Dim3(strides), num_per_cell);
                break;
        }
    }

    return error_code;
}
