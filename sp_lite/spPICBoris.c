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
#include "spRandom.h"

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

int spBorisYeeParticleInitialize(spParticle *sp, Real n0, Real T0, size_type num_pic)
{
    SP_CALL(spParticleDeploy(sp));

    size_type max_number_of_entities = spParticleGetNumberOfEntities(sp);

    int dist_type[6] = {SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_NORMAL, SP_RAND_NORMAL, SP_RAND_NORMAL};

    SP_CALL(spParticleInitialize(sp, num_pic, dist_type));

    boris_particle *data = (boris_particle *) spParticleGetDeviceData(sp);
//    Real *v[3] = {data->vx, data->vx, data->vx};
//    Real u[3] = {0, 0, 0};
//    SP_CALL(spRandomUniformNormal6(v, max_number_of_entities, u, sqrt(2.0 * T0 * SI_Boltzmann_constant / spParticleGetMass(sp))));
    SP_CALL(spParallelDeviceFillReal(data->f, n0, max_number_of_entities));

    SP_CALL(spParallelMemset(data->w, 0, max_number_of_entities * sizeof(Real)));
    return SP_SUCCESS;
}
