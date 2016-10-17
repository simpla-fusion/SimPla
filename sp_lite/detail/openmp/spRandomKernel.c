//
// Created by salmon on 16-8-14.
//


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../../spRandom.h"
#include "../../spParallel.h"
#include "../../sp_lite_config.h"

/* Number of 64-bit vectors per dimension */
#define VECTOR_SIZE 64


typedef struct spRandomGenerator_s
{
    int num_of_dimensions;

} spRandomGenerator;

int spRandomGeneratorCreate(spRandomGenerator **gen, int type, int num_of_dimension, size_type offset)
{

    SP_CALL(spMemoryHostAlloc((void **) gen, sizeof(spRandomGenerator)));

    SP_CALL(spRandomGeneratorSetNumOfDimensions(*gen, num_of_dimension));

    return SP_SUCCESS;
}

int spRandomGeneratorDestroy(spRandomGenerator **gen)
{

    SP_CALL(spMemoryHostFree((void **) gen));

    return SP_SUCCESS;
}

int spRandomGeneratorSetNumOfDimensions(spRandomGenerator *gen, int n)
{
    gen->num_of_dimensions = n;
    return SP_SUCCESS;
}

int spRandomGeneratorGetNumOfDimensions(spRandomGenerator const *gen) { return gen->num_of_dimensions; }

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

//    int n_dims = spRandomGeneratorGetNumOfDimensions(gen);
//    for (int n = 0; n < n_dims; ++n)
//    {
//        switch (dist_types[n])
//        {
//            case SP_RAND_UNIFORM:
//
//                return SP_UNIMPLEMENTED;
//
//            case SP_RAND_NORMAL:
//            default:
//
//                break;
//        }
//    }
    spRandomMultiNormalDistributionInCell(min,
                                          max,
                                          strides,
                                          (unsigned int) num_per_cell,
                                          data[0],
                                          data[1],
                                          data[2],
                                          data[3],
                                          data[4],
                                          data[5]);


    return SP_SUCCESS;
}
