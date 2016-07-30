//
// Created by salmon on 16-7-30.
//

#ifndef SIMPLA_SPRANDOM_H
#define SIMPLA_SPRANDOM_H

#include "sp_lite_def.h"

struct spRandomGenerator_s;

typedef struct spRandomGenerator_s spRandomGenerator;

enum { SP_RAND_UNIFORM = 0, SP_RAND_SOBOL };

int spRandomGeneratorCreate(spRandomGenerator **gen, int type, int num_of_dimension, size_type offset);

int spRandomGeneratorDestroy(spRandomGenerator **gen);

int spRandomGenerateSimple(spRandomGenerator *gen, Real **data, size_type num_of_sample, int num_of_dimension,
                           int const *dist_types, Real const *a, Real const *b);

int spRandomUniformNormal(spRandomGenerator *gen, Real **data, size_type num_of_sample, int num_of_dimension,
                          Real const *a, Real const *b);

#endif //SIMPLA_SPRANDOM_H
