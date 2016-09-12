//
// Created by salmon on 16-7-30.
//

#ifndef SIMPLA_SPRANDOM_H
#define SIMPLA_SPRANDOM_H

#include "sp_config.h"


struct spRandomGenerator_s;

typedef struct spRandomGenerator_s spRandomGenerator;

enum { SP_RAND_GEN_SOBOL };
enum { SP_RAND_UNIFORM = 0x1, SP_RAND_NORMAL = 0x10 };

/**
 *  \f[
 *      f\left(v\right)\equiv\frac{1}{\sqrt{\left(2\pi\sigma\right)^{3}}}\exp\left(-\frac{\left(v-u\right)^{2}}{\sigma^{2}}\right)
 *  \f]
 */
int spRandomGeneratorCreate(spRandomGenerator **gen, int type, int num_of_dimension, int offset);

int spRandomGeneratorDestroy(spRandomGenerator **gen);

int spRandomGeneratorSetNumOfDimensions(spRandomGenerator *gen, int n);

int spRandomGeneratorGetNumOfDimensions(spRandomGenerator const *gen);

//int spRandomGeneratorSetThreadBlocks(spRandomGenerator *gen, int const *blocks, int const *threads);
//
//int spRandomGeneratorGetThreadBlocks(spRandomGenerator *gen, int *blocks, int *threads);

int spRandomGeneratorGetNumOfThreads(spRandomGenerator const *gen);

int spRandomMultiDistributionInCell(spRandomGenerator *gen, int const *dist_types, Real **data,
                                    int const *min, int const *max, int const *strides,
                                    int num_per_cell);
int spRandomMultiNormalDistributionInCell(int const *min,
                                          int const *max,
                                          int const *strides,
                                          unsigned int pic,
                                          Real *rx,
                                          Real *ry,
                                          Real *rz,
                                          Real *vx,
                                          Real *vy,
                                          Real *vz);

#endif //SIMPLA_SPRANDOM_H
