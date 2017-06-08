//
// Created by salmon on 17-6-8.
//
#include "Particle.h"
#include "spParticle.h"
namespace simpla {

void ParticleInitialLoad(Rng* seed_rng, void* p) {
    auto* sp = reinterpret_cast<spParticle_s*>(p);

    for (int i = 0; i < 3; ++i) {
        RandomDistributionUniform(seed_rng[i], sp->r[i], sp->max_size);
        RandomDistributionNormal(seed_rng[i + 3], sp->v[i], sp->max_size);
    }

    memset(sp->tag, 0, sp->max_size * sizeof(int));

    for (int j = 6; j < sp->dof; ++j) { memset(sp->f[j - 6], 0, sp->max_size * sizeof(Real)); }
}

}  // namespace simpla{
