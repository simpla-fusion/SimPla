//
// Created by salmon on 16-7-20.
//

#include "sp_lite_def.h"

#include "../src/sp_capi.h"

#include "BorisYee.h"
#include "spMesh.h"
#include "spField.h"

int spBorisYeeParticleCreate(spParticle **sp, struct spMesh_s const *m)
{
    if (sp == NULL) { return SP_FAILED; }

    SP_PARTICLE_CREATE_DATA_DESC(data_desc, struct boris_particle_s);
    SP_PARTICLE_CREATE_DATA_DESC_ADD(data_desc, struct boris_particle_s, Real, vx);
    SP_PARTICLE_CREATE_DATA_DESC_ADD(data_desc, struct boris_particle_s, Real, vy);
    SP_PARTICLE_CREATE_DATA_DESC_ADD(data_desc, struct boris_particle_s, Real, vz);
    SP_PARTICLE_CREATE_DATA_DESC_ADD(data_desc, struct boris_particle_s, Real, f);
    SP_PARTICLE_CREATE_DATA_DESC_ADD(data_desc, struct boris_particle_s, Real, w);


//    LOAD_KERNEL(spBorisInitializeParticleKernel,
//                sizeType2Dim3(spMeshGetShape(spParticleMesh(sp))),
//                NUMBER_OF_THREADS_PER_BLOCK,
//                (boris_particle *) spParticleAttributeDeviceData(sp),
//                spParticleBuckets(sp),
//                spParticlePagePool(sp),
//                NUM_OF_PIC);
//	spUpdateParticleBorisScatterBlockKernel<<< sp->m->topology_dims, NUMBER_OF_THREADS_PER_BLOCK >>>(sp->buckets,
//			(fRho->device_data), ( fJ->device_data));

    spParticleCreate(sp, m, data_desc);

    spDataTypeDestroy(&data_desc);

    return SP_SUCCESS;
}
int spUpdateField_Yee(struct spMesh_s const *m,
                      Real dt,
                      const struct spField_s *fRho,
                      const struct spField_s *fJ,
                      struct spField_s *fE,
                      struct spField_s *fB)
{
    if (m == NULL) { return SP_FAILED; }

    return SP_SUCCESS;
}
