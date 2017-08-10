//
// Created by salmon on 17-8-9.
//

#ifndef SIMPLA_PARTICLEINITIALLOAD_H
#define SIMPLA_PARTICLEINITIALLOAD_H

#include "simpla/SIMPLA_config.h"

namespace simpla {
enum { SP_RAND_UNIFORM = 0x1, SP_RAND_NORMAL = 0x10 };

int ParticleInitialLoad(Real **, size_type num, int n_dof, int const *dist_types, size_type random_seed_offset);
}  // namespace simpla{

#endif  // SIMPLA_PARTICLEINITIALLOAD_H
