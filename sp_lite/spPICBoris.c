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


int spBorisYeeParticleCreate(spParticle **sp, struct spMesh_s const *m)
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

int spBorisYeeParticleInitialize(spParticle *sp, Real n0, Real T0)
{
    SP_CALL(spParticleDeploy(sp));

    size_type number_of_entities = spParticleNumberOfEntities(sp);

    spParticleInitialize(sp);

    struct boris_particle_s *data = (struct boris_particle_s *) spParticleData(sp);

    Real *v[3] = {data->vx, data->vx, data->vx};

    Real u[3] = {0, 0, 0};

    spRandomNormal3(v, number_of_entities, u, sqrt(2.0 * T0 * SI_Boltzmann_constant / spParticleGetMass(sp)));

    spParallelMemset(data->f, 0, number_of_entities * sizeof(Real));

    spParallelMemset(data->w, 0, number_of_entities * sizeof(Real));

    return SP_SUCCESS;
}
