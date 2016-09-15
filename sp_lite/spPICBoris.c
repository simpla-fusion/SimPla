//
// Created by salmon on 16-7-20.
//
#include <math.h>

#include "spPICBoris.h"
#include "sp_lite_def.h"
#include "spDataType.h"
#include "spParallel.h"
#include "spPhysicalConstants.h"


#include "spMesh.h"
#include "spField.h"

int spParticleCreateBorisYee(spParticle **sp, struct spMesh_s const *m)
{
    if (sp == NULL) { return SP_DO_NOTHING; }
    int error_code = SP_SUCCESS;
    SP_CALL(spParticleCreate(sp, m));
    SP_PARTICLE_CREATE_DATA_DESC((*sp), struct boris_particle_s);
    SP_PARTICLE_ADD_ATTR((*sp), vx);
    SP_PARTICLE_ADD_ATTR((*sp), vy);
    SP_PARTICLE_ADD_ATTR((*sp), vz);
    SP_PARTICLE_ADD_ATTR((*sp), f);
    SP_PARTICLE_ADD_ATTR((*sp), w);
    return error_code;
}

int spParticleDestroyBorisYee(spParticle **sp)
{
    return spParticleDestroy(sp);
}


