//
// Created by salmon on 16-7-28.
//



#include <stdio.h>
#include <stdlib.h>
#include "spParallelCUDA.h"
//
//
//__global__ void setup_kernel(curandStateSobol32_t *state)
//{
//    int id = threadIdx.x + blockIdx.x * 64;
//    /* Each thread gets same seed, a different sequence
//       number, no offset */
//    curand_init(1234, id, 0, &state[id]);
//}
//
//
//__global__ void generate_uniform_kernel(curandStateSobol32_t *state,
//                                        int n,
//                                        unsigned int *result)
//{
//    int id = threadIdx.x + blockIdx.x * 64;
//    unsigned int count = 0;
//    float x;
//    /* Copy state to local memory for efficiency */
//    curandStateSobol32_t localState = state[id];
//    /* Generate pseudo-random uniforms */
//    for (int i = 0; i < n; i++)
//    {
//        x = curand_uniform(&localState);
//        /* Check if > .5 */
//        if (x > .5)
//        {
//            count++;
//        }
//    }
//    /* Copy state back to global memory */
//    state[id] = localState;
//    /* Store results */
//    result[id] += count;
//}
//
//
//__global__ void generate_normal_kernel(curandStateSobol32_t *state,
//                                       int n,
//                                       unsigned int *result)
//{
//    int id = threadIdx.x + blockIdx.x * 64;
//    unsigned int count = 0;
//    float2 x;
//    /* Copy state to local memory for efficiency */
//    curandStateSobol32_t localState = state[id];
//    /* Generate pseudo-random normals */
//    for (int i = 0; i < n / 2; i++)
//    {
//        x = curand_normal2(&localState);
//        /* Check if within one standard deviaton */
//        if ((x.x > -1.0) && (x.x < 1.0))
//        {
//            count++;
//        }
//        if ((x.y > -1.0) && (x.y < 1.0))
//        {
//            count++;
//        }
//    }
//    /* Copy state back to global memory */
//    state[id] = localState;
//    /* Store results */
//    result[id] += count;
//}

//int main(int argc, char *argv[])
//{
//
//    curandStateSobol32_t *devStates;
//
//    unsigned int *devResults, *hostResults;
//
//    int sampleCount = 10000;
//
//
//    /* @formatter:off */
//    setup_kernel <<< 64, 64 >>> (devStates);
//    generate_uniform_kernel <<< 64, 64 >>> (devStates, sampleCount, devResults);
//    generate_normal_kernel <<< 64, 64 >>> (devStates, sampleCount, devResults);
//    /* @formatter:on */
//
//    CUDA_CALL(cudaFree(devStates));
//    CUDA_CALL(cudaFree(devResults));
//
//}