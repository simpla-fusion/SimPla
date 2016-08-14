//
// Created by salmon on 16-8-14.
//

#include "../../spPICBoris.h"
#include "../../spMesh.h"
#include "../../spRandom.h"
#include "../../spPhysicalConstants.h"
#include "../../spField.h"
#include <math.h>

int spParticleInitializeBorisYee(spParticle *sp, Real n0, Real T0, int do_import_sample)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    SP_CALL(spParticleDeploy(sp));

    size_type max_number_of_entities = spParticleGetNumberOfEntities(sp);

    int dist_type[6] = {SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_NORMAL, SP_RAND_NORMAL, SP_RAND_NORMAL};

    SP_CALL(spParticleInitialize(sp, dist_type));

    Real dx[3];
    SP_CALL(spMeshGetDx(m, dx));
    Real vT = (Real) sqrt(2.0 * SI_Boltzmann_constant * T0 / spParticleGetMass(sp));
    Real f0 = n0 * dx[0] * dx[1] * dx[2] / spParticleGetPIC(sp);
    size_type x_min[3], x_max[3], strides[3];
    SP_CALL(spMeshGetArrayShape(m, SP_DOMAIN_CENTER, x_min, x_max, strides));
    strides[0] *= spParticleGetMaxPIC(sp);
    strides[1] *= spParticleGetMaxPIC(sp);
    strides[2] *= spParticleGetMaxPIC(sp);

    size_type blocks[3] = {16, 1, 1};
    size_type threads[3] = {SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE, 1, 1};

    void **device_data;

    spParticleGetAllAttributeData_device(sp, &device_data);

//    SP_DEVICE_CALL_KERNEL(spParticleInitializeBorisYeeKernel,
//                          sizeType2Dim3(blocks), sizeType2Dim3(threads),
//                          (boris_particle *) m_data_,
//                          sizeType2Dim3(x_min), sizeType2Dim3(x_max), sizeType2Dim3(strides),
//                          spParticleGetPIC(sp), vT, f0, do_import_sample
//    );

//    spParallelDeviceFree((void **) &m_data_);
    return SP_SUCCESS;
}

int spParticleUpdateBorisYee(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho, spField *fJ)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

//    boris_update_param update_param;
//    update_param.max_pic = (int) spParticleGetMaxPIC(sp);
//    update_param.cmr_dt = dt * spParticleGetCharge(sp) / spParticleGetMass(sp);
//    size_type min[3], max[3], strides[3];
//    SP_CALL(spMeshGetInvDx(m, update_param.inv_dv));
//    SP_CALL(spMeshGetArrayShape(m, SP_DOMAIN_ALL, min, max, strides));
//    for (int i = 0; i < 3; ++i)
//    {
//        update_param.inv_dv[i] *= dt;
//        update_param.min[i] = (int) min[i];
//        update_param.max[i] = (int) max[i];
//        update_param.strides[i] = (int) strides[i];
//    }
//
//    size_type field_size = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, spMeshAttributeGetForm((spMeshAttribute const *) fE));
//
//    SP_CALL(spParticleGetAllAttributeData(sp, update_param.data));
//    SP_CALL(spFieldSubArray(fRho, (void **) &update_param.rho));
//    SP_CALL(spFieldSubArray(fJ, (void **) update_param.J));
//    SP_CALL(spFieldSubArray((spField *) fE, (void **) update_param.E));
//    SP_CALL(spFieldSubArray((spField *) fB, (void **) update_param.B));
    /**

     Update Here

     */

    SP_CALL(spParticleSync(sp));
    SP_CALL(spFieldSync(fJ));
    SP_CALL(spFieldSync(fRho));
    return SP_SUCCESS;
}
