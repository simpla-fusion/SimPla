//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_BORISYEE_H
#define SIMPLA_BORISYEE_H

#include "spParticle.h"
#include "spMesh.h"
#include "spField.h"


typedef struct boris_particle_s
{
    SP_PARTICLE_HEAD
    SP_PARTICLE_ATTR(Real, vx)
    SP_PARTICLE_ATTR(Real, vy)
    SP_PARTICLE_ATTR(Real, vz)
    SP_PARTICLE_ATTR(Real, f)
    SP_PARTICLE_ATTR(Real, w)

} boris_particle;

int spBorisYeeParticleCreate(spMesh const *m, spParticle **sp);

int spBorisYeeParticleUpdate(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho,
                             spField *fJ);

int spUpdateField_Yee(spMesh *ctx, Real dt, const spField *fRho, const spField *fJ, spField *fE, spField *fB);

#endif //SIMPLA_BORISYEE_H
