//
// Created by salmon on 16-7-20.
//
#include <math.h>

#include "spPICBoris.h"
#include "sp_lite_config.h"
#include "spDataType.h"
#include "spParallel.h"
#include "spPhysicalConstants.h"


#include "spMesh.h"
#include "spField.h"

int spParticleCreateBorisYee(spParticle **sp, struct spMesh_s const *m)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    SP_CALL(spParticleCreate(sp, m));
    SP_PARTICLE_ATTR_HEAD((*sp));
    SP_PARTICLE_ATTR_ADD((*sp), vx);
    SP_PARTICLE_ATTR_ADD((*sp), vy);
    SP_PARTICLE_ATTR_ADD((*sp), vz);
    SP_PARTICLE_ATTR_ADD((*sp), f);
    SP_PARTICLE_ATTR_ADD((*sp), w);

    return SP_SUCCESS;
}

int spParticleDestroyBorisYee(spParticle **sp)
{
    SP_CALL(spParticleDestroy(sp));
    return SP_SUCCESS;
}


