//
// Created by salmon on 16-7-20.
//

#include "sp_lite_def.h"

#include "../src/sp_capi.h"

#include "FDTDBoris.h"
#include "spMesh.h"
#include "spField.h"

int spBorisYeeParticleCreate(spParticle **sp, struct spMesh_s const *m)
{
    if (sp == NULL) { return SP_FAILED; }

    SP_CHECK_RETURN(spParticleCreate(sp, m));


    SP_PARTICLE_CREATE_DATA_DESC((*sp), struct boris_particle_s);
    SP_PARTICLE_DATA_DESC_ADD((*sp), struct boris_particle_s, Real, vx);
    SP_PARTICLE_DATA_DESC_ADD((*sp), struct boris_particle_s, Real, vy);
    SP_PARTICLE_DATA_DESC_ADD((*sp), struct boris_particle_s, Real, vz);
    SP_PARTICLE_DATA_DESC_ADD((*sp), struct boris_particle_s, Real, f);
    SP_PARTICLE_DATA_DESC_ADD((*sp), struct boris_particle_s, Real, w);


//    LOAD_KERNEL(spBorisInitializeParticleKernel,
//                sizeType2Dim3(spMeshArrayShape(spParticleMesh(sp))),
//                NUMBER_OF_THREADS_PER_BLOCK,
//                (boris_particle *) spParticleAttributeDeviceData(sp),
//                spParticleBuckets(sp),
//                spParticlePagePool(sp),
//                NUM_OF_PIC);
//	spUpdateParticleBorisScatterBlockKernel<<< sp->m->topology_dims, NUMBER_OF_THREADS_PER_BLOCK >>>(sp->buckets,
//			(fRho->device_data), ( fJ->device_data));
    return SP_SUCCESS;
}
