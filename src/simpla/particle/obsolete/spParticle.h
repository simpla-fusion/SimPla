//
// Created by salmon on 17-6-8.
//

#ifndef SIMPLA_SPPARTICLE_H
#define SIMPLA_SPPARTICLE_H

#include "simpla/SIMPLA_config.h"

#define SP_MAX_SIZE_IN_CELL 256

struct spParticle_s {
    size_type size;
    int dof;
    int* tag;
    Real* r[3];
    Real* v[3];
    Real* f[];
};

#endif  // SIMPLA_SPPARTICLE_H
