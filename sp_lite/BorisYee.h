//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_BORISYEE_H
#define SIMPLA_BORISYEE_H

#include "sp_lite_def.h"
#include "spParticle.h"


typedef struct boris_particle_s
{
    SP_PARTICLE_HEAD
    SP_PARTICLE_ATTR(Real, vx)
    SP_PARTICLE_ATTR(Real, vy)
    SP_PARTICLE_ATTR(Real, vz)
    SP_PARTICLE_ATTR(Real, f)
    SP_PARTICLE_ATTR(Real, w)

} boris_particle;

struct spMesh_s;
struct spField_s;

int spBorisYeeParticleCreate(spParticle **sp, struct spMesh_s const *m);

int spBorisYeeParticleUpdate(spParticle *sp,
                             Real dt,
                             const struct spField_s *fE,
                             const struct spField_s *fB,
                             struct spField_s *fRho,
                             struct spField_s *fJ);

int spUpdateField_Yee(struct spMesh_s const *ctx,
                      Real dt,
                      const struct spField_s *fRho,
                      const struct spField_s *fJ,
                      struct spField_s *fE,
                      struct spField_s *fB);

#endif //SIMPLA_BORISYEE_H
