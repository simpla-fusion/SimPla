//
// Created by salmon on 16-8-14.
//


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../../spRandom.h"
#include "../../spParallel.h"
/* Number of 64-bit vectors per dimension */
#define VECTOR_SIZE 64


typedef struct spRandomGenerator_s
{
    int num_of_dimensions;
    size_type blocks[3];
    size_type threads[3];
    size_type num_of_threads;
} spRandomGenerator;


int spRandomGeneratorCreate(spRandomGenerator **gen, int type, int num_of_dimension, size_type offset)
{

    SP_CALL(spParallelHostAlloc((void **) gen, sizeof(spRandomGenerator)));
    {
        size_type blocks[3] = {16, 1, 1};
        size_type threads[3] = {64, 1, 1};
        spRandomGeneratorSetThreadBlocks(*gen, blocks, threads);
        spRandomGeneratorSetNumOfDimensions(*gen, num_of_dimension);
    }
    int n_dims = spRandomGeneratorGetNumOfDimensions(*gen);
    size_type n_threads = spRandomGeneratorGetNumOfThreads(*gen);


    return SP_SUCCESS;
}

int spRandomGeneratorDestroy(spRandomGenerator **gen)
{

    return spParallelHostFree((void **) gen);
}

int spRandomGeneratorSetNumOfDimensions(spRandomGenerator *gen, int n)
{
    gen->num_of_dimensions = n;
    return SP_SUCCESS;
}

int spRandomGeneratorGetNumOfDimensions(spRandomGenerator const *gen) { return gen->num_of_dimensions; }

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

size_type spRandomGeneratorGetNumOfThreads(spRandomGenerator const *gen) { return gen->num_of_threads; }


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
    size_type s_blocks[3], s_threads[3];
    spRandomGeneratorGetThreadBlocks(gen, s_blocks, s_threads);

    int n_dims = spRandomGeneratorGetNumOfDimensions(gen);
    size_type n_threads = spRandomGeneratorGetNumOfThreads(gen);
    for (int n = 0; n < n_dims; ++n)
    {
        switch (dist_types[n])
        {
            case SP_RAND_NORMAL:

//                SP_DEVICE_CALL_KERNEL(spRandomDistributionInCellNormalKernel,
//                                      sizeType2Dim3(s_blocks), sizeType2Dim3(s_threads),
//                                      gen->devSobol64States + n * n_threads,
//                                      data[n],
//                                      sizeType2Dim3(min), sizeType2Dim3(max), sizeType2Dim3(strides), num_per_cell);
                break;
            case SP_RAND_UNIFORM:
            default:
//                SP_DEVICE_CALL_KERNEL(spRandomDistributionInCellUniformKernel,
//                                      sizeType2Dim3(s_blocks), sizeType2Dim3(s_threads),
//                                      gen->devSobol64States + n * n_threads,
//                                      data[n],
//                                      sizeType2Dim3(min), sizeType2Dim3(max), sizeType2Dim3(strides), num_per_cell);
                break;
        }
    }

    return SP_SUCCESS;
}
