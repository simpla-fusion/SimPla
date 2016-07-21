//
// Created by salmon on 16-7-20.
//



#include "BorisYee.h"

void spBorisYeeInitializeParticle(spParticle *sp, size_type NUM_OF_PIC)
{
//    ADD_PARTICLE_ATTRIBUTE(sp, Real, vx);
//    ADD_PARTICLE_ATTRIBUTE(sp, Real, vy);
//    ADD_PARTICLE_ATTRIBUTE(sp, Real, vz);
//    ADD_PARTICLE_ATTRIBUTE(sp, Real, f);
//    ADD_PARTICLE_ATTRIBUTE(sp, Real, w);
//
//    spParticleDeploy(sp, NUM_OF_PIC);


//    LOAD_KERNEL(spBorisInitializeParticleKernel,
//                sizeType2Dim3(spMeshGetShape(spParticleMesh(sp))),
//                NUMBER_OF_THREADS_PER_BLOCK,
//                (boris_data *) spParticleAttributeDeviceData(sp),
//                spParticleBuckets(sp),
//                spParticlePagePool(sp),
//                NUM_OF_PIC);


//	spUpdateParticleBorisScatterBlockKernel<<< sp->m->topology_dims, NUMBER_OF_THREADS_PER_BLOCK >>>(sp->buckets,
//			(fRho->device_data), ( fJ->device_data));

    spParallelDeviceSync();        //wait for iteration to finish
    spParticleSync(sp);
    spParallelDeviceSync();        //wait for iteration to finish


}

void spBorisYeeUpdateParticle(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho,
                              spField *fJ)
{

};
void spUpdateField_Yee(spMesh *ctx, Real dt, const spField *fRho, const spField *fJ, spField *fE, spField *fB)
{

}
