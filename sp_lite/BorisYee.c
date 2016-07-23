//
// Created by salmon on 16-7-20.
//



#include "BorisYee.h"

int spBorisYeeParticleCreate(spMesh const *m, spParticle **sp)
{
    if (sp == NULL) { return SP_FAILED; }


    spParticleCreate(m, sp);

    // particle page head

    SP_PARTICLE_ADD_ATTR_HEAD((*sp), boris_particle)
    SP_PARTICLE_ADD_ATTR((*sp), boris_particle, Real, vz);
    SP_PARTICLE_ADD_ATTR((*sp), boris_particle, Real, vz);
    SP_PARTICLE_ADD_ATTR((*sp), boris_particle, Real, vz);
    SP_PARTICLE_ADD_ATTR((*sp), boris_particle, Real, f);
    SP_PARTICLE_ADD_ATTR((*sp), boris_particle, Real, w);

//    LOAD_KERNEL(spBorisInitializeParticleKernel,
//                sizeType2Dim3(spMeshGetShape(spParticleMesh(sp))),
//                NUMBER_OF_THREADS_PER_BLOCK,
//                (boris_particle *) spParticleAttributeDeviceData(sp),
//                spParticleBuckets(sp),
//                spParticlePagePool(sp),
//                NUM_OF_PIC);


//	spUpdateParticleBorisScatterBlockKernel<<< sp->m->topology_dims, NUMBER_OF_THREADS_PER_BLOCK >>>(sp->buckets,
//			(fRho->device_data), ( fJ->device_data));

    return SP_SUCCESS;
}

int spBorisYeeParticleUpdate(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho,
                             spField *fJ)
{
    if (sp == NULL) { return SP_FAILED; }

    return SP_SUCCESS;
};

int spUpdateField_Yee(spMesh *ctx, Real dt, const spField *fRho, const spField *fJ, spField *fE, spField *fB)
{
    if (ctx == NULL) { return SP_FAILED; }

    return SP_SUCCESS;
}
