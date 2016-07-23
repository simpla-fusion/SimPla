//
// Created by salmon on 16-7-20.
//



#include "BorisYee.h"

void spBorisYeeParticleCreate(spParticle **sp, spMesh const *m, size_type NUM_OF_PIC)
{
    spParticleCreate(sp, m);

    // particle page head

    SP_PARTICLE_ATTR_HEAD((*sp), boris_particle)
    SP_PARTICLE_ADD_ATTR((*sp), boris_particle, Real, vz);
    SP_PARTICLE_ADD_ATTR((*sp), boris_particle, Real, vz);
    SP_PARTICLE_ADD_ATTR((*sp), boris_particle, Real, vz);
    SP_PARTICLE_ADD_ATTR((*sp), boris_particle, Real, f);
    SP_PARTICLE_ADD_ATTR((*sp), boris_particle, Real, w);


    spParticleDeploy(*sp, NUM_OF_PIC);


//    LOAD_KERNEL(spBorisInitializeParticleKernel,
//                sizeType2Dim3(spMeshGetShape(spParticleMesh(sp))),
//                NUMBER_OF_THREADS_PER_BLOCK,
//                (boris_particle *) spParticleAttributeDeviceData(sp),
//                spParticleBuckets(sp),
//                spParticlePagePool(sp),
//                NUM_OF_PIC);


//	spUpdateParticleBorisScatterBlockKernel<<< sp->m->topology_dims, NUMBER_OF_THREADS_PER_BLOCK >>>(sp->buckets,
//			(fRho->device_data), ( fJ->device_data));



}

void spBorisYeeParticleUpdate(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho,
                              spField *fJ)
{

};
void spUpdateField_Yee(spMesh *ctx, Real dt, const spField *fRho, const spField *fJ, spField *fE, spField *fB)
{

}
