//
// Created by salmon on 16-7-20.
//

#include "sp_lite_def.h"
#include "spPICBoris.h"

#include <math.h>

#include "../src/sp_capi.h"

#include "spMesh.h"
#include "spField.h"
#include "spParallel.h"
#include "spPhysicalConstants.h"

int spParticleCreateBorisYee(spParticle **sp, struct spMesh_s const *m)
{
    if (sp == NULL) { return SP_FAILED; }

    SP_CALL(spParticleCreate(sp, m));
    SP_PARTICLE_CREATE_DATA_DESC((*sp), struct boris_particle_s);
    SP_PARTICLE_DATA_DESC_ADD((*sp), struct boris_particle_s, Real, vx);
    SP_PARTICLE_DATA_DESC_ADD((*sp), struct boris_particle_s, Real, vy);
    SP_PARTICLE_DATA_DESC_ADD((*sp), struct boris_particle_s, Real, vz);
    SP_PARTICLE_DATA_DESC_ADD((*sp), struct boris_particle_s, Real, f);
    SP_PARTICLE_DATA_DESC_ADD((*sp), struct boris_particle_s, Real, w);

}

int spParticleDestroyBorisYee(spParticle **sp) { return spParticleDestroy(sp); }


