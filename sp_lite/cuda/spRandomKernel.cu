//
// Created by salmon on 16-7-28.
//
extern "C"
{
#include <stdio.h>
#include <stdlib.h>
#include "spParallelCUDA.h"

#include </usr/local/cuda/include/curand.h>
}


#define THREADS_PER_BLOCK 64
#define BLOCK_COUNT 64
#define TOTAL_THREADS (THREADS_PER_BLOCK * BLOCK_COUNT)

/* Number of 64-bit vectors per dimension */
#define VECTOR_SIZE 64

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)



/**
 * This kernel initializes state per thread for each of x, y, and z,vx,vy,vz
 */
__global__ void
setup_kernel(unsigned long long *sobolDirectionVectors,
             unsigned long long *sobolScrambleConstants,
             int num_of_dim, size_type offset,
             curandStateScrambledSobol64 *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int dim = num_of_dim * id;
    /* Each thread uses 3 different dimensions */
    for (int i = 0; i < num_of_dim; ++i)
    {
        curand_init(sobolDirectionVectors + VECTOR_SIZE * (dim + i),
                    sobolScrambleConstants[dim + i],
                    offset,
                    &state[dim + i]);
    }

}

/**
 * This kernel generates random 6D points and increments a counter if
 * a point is within a unit sphere   
 *  \f[
 *      f\left(v\right)\equiv\frac{1}{\sqrt{\left(2\pi\sigma\right)^{3}}}\exp\left(-\frac{\left(v-u\right)^{2}}{\sigma^{2}}\right)
 *  \f]
 * @param data
 * @param num_of_sample
 * @param u0
 * @param sigma
 * @return
 */

__global__ void
generate_kernel(curandStateScrambledSobol64 *state,
                Real **data, size_type offset, size_type num, Real3 min, Real3 length, Real3 u0, Real vT)
{
    size_type id = threadIdx.x + blockIdx.x * blockDim.x;
    size_type num_of_thread = blockDim.x * gridDim.x;

    /* Generate quasi-random double precision coordinates */
    for (size_type s = id; s < num; s += num_of_thread)
    {

        data[0][s + offset] = curand_uniform(&state[id * 6 + 0]) * length.x + min.x;
        data[1][s + offset] = curand_uniform(&state[id * 6 + 1]) * length.y + min.y;
        data[2][s + offset] = curand_uniform(&state[id * 6 + 2]) * length.z + min.z;

        data[3][s + offset] = curand_normal(&state[id * 6 + 3]) * vT + u0.x;
        data[4][s + offset] = curand_normal(&state[id * 6 + 4]) * vT + u0.y;
        data[5][s + offset] = curand_normal(&state[id * 6 + 5]) * vT + u0.z;
    }

}

typedef struct spRandomSobolSequences_s
{
    curandStateScrambledSobol64 *devSobol64States;

    unsigned long long int *devDirectionVectors64;
    unsigned long long int *devScrambleConstants64;
    long long int *devResults, *hostResults;
} spRandomSobolSequences;

int spRandomSobolCreate(spRandomSobolSequences **gen, int num_of_dimension, size_type offset)
{
    SP_CALL(spParallelHostAlloc((void **) gen, sizeof(spRandomSobolSequences)));


    curandDirectionVectors64_t *hostVectors64;
    unsigned long long int *hostScrambleConstants64;

    /* Get pointers to the 64 bit scrambled direction vectors and constants*/
    CURAND_CALL(curandGetDirectionVectors64(&hostVectors64,
                                            CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));

    CURAND_CALL(curandGetScrambleConstants64(&hostScrambleConstants64));


    /* Allocate memory for 3 states per thread (x, y, z), each state to get a unique dimension */
    CUDA_CALL(cudaMalloc((void **) &((*gen)->devSobol64States),
                         TOTAL_THREADS * 6 * sizeof(curandStateScrambledSobol64)));

    /* Allocate memory and copy 3 sets of vectors per thread to the device */

    CUDA_CALL(cudaMalloc((void **) &((*gen)->devDirectionVectors64),
                         6 * TOTAL_THREADS * VECTOR_SIZE * sizeof(long long int)));

    CUDA_CALL(cudaMemcpy((*gen)->devDirectionVectors64, hostVectors64,
                         6 * TOTAL_THREADS * VECTOR_SIZE * sizeof(long long int),
                         cudaMemcpyHostToDevice));

    /* Allocate memory and copy 6 scramble constants (one costant per dimension)
       per thread to the device */

    CUDA_CALL(cudaMalloc((void **) &((*gen)->devScrambleConstants64),
                         6 * TOTAL_THREADS * sizeof(long long int)));

    CUDA_CALL(cudaMemcpy((*gen)->devScrambleConstants64, hostScrambleConstants64,
                         6 * TOTAL_THREADS * sizeof(long long int),
                         cudaMemcpyHostToDevice));
    /* @formatter:off */
    /* Initialize the states */

    setup_kernel<<<BLOCK_COUNT, THREADS_PER_BLOCK>>>((*gen)->devDirectionVectors64,
       (*gen)-> devScrambleConstants64,
        num_of_dimension,offset,
       (*gen)-> devSobol64States
    );
    /* @formatter:on */

    return SP_SUCCESS;
}

int spRandomSobolDestroy(spRandomSobolSequences **gen)
{
    CUDA_CALL(cudaFree((*gen)->devSobol64States));
    CUDA_CALL(cudaFree((*gen)->devDirectionVectors64));
    CUDA_CALL(cudaFree((*gen)->devScrambleConstants64));


    return spParallelHostFree((void **) gen);
}
int
spRandomUniformNormal6(spRandomSobolSequences *gen, Real **data, size_type num_of_sample, Real const *u0, Real sigma)
{
    /* @formatter:off */
    generate_kernel<<<BLOCK_COUNT, THREADS_PER_BLOCK>>>(gen->devSobol64States, num_of_sample, data);
    /* @formatter:on */
    return EXIT_SUCCESS;
}