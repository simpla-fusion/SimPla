//
// Created by salmon on 16-7-30.
//

#ifndef SIMPLA_SPRANDOM_H
#define SIMPLA_SPRANDOM_H

#include "sp_lite_def.h"

struct spRandomGenerator_s;

typedef struct spRandomGenerator_s spRandomGenerator;
enum { SP_RAND_GEN_SOBOL };
enum { SP_RAND_UNIFORM = 0x1, SP_RAND_NORMAL = 0x10 };

/**
 *  \f[
 *      f\left(v\right)\equiv\frac{1}{\sqrt{\left(2\pi\sigma\right)^{3}}}\exp\left(-\frac{\left(v-u\right)^{2}}{\sigma^{2}}\right)
 *  \f]
 */
int spRandomGeneratorCreate(spRandomGenerator **gen, int type, int num_of_dimension, size_type offset);

int spRandomGeneratorDestroy(spRandomGenerator **gen);

int spRandomGeneratorSetNumOfDimensions(spRandomGenerator *gen, int n);

int spRandomGeneratorGetNumOfDimensions(spRandomGenerator const *gen);

//int spRandomGeneratorSetThreadBlocks(spRandomGenerator *gen, size_type const *blocks, size_type const *threads);
//
//int spRandomGeneratorGetThreadBlocks(spRandomGenerator *gen, size_type *blocks, size_type *threads);

size_type spRandomGeneratorGetNumOfThreads(spRandomGenerator const *gen);

int spRandomMultiDistributionInCell(spRandomGenerator *gen, int const *dist_types, Real **data,
                                    size_type const *min, size_type const *max, size_type const *strides,
                                    size_type num_per_cell);


#endif //SIMPLA_SPRANDOM_H
