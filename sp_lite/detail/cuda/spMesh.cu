//
// Created by salmon on 16-9-24.
//

#include "../spMesh.cu.h"

__constant__
_spMeshDevice _sp_mesh;

int spMeshSetupParam(spMesh *m)
{
    int error_code = SP_SUCCESS;
    _spMeshDevice param;
    size_type strides[3], dims[3];

    int iform = VERTEX;


    SP_CALL(spMeshGetDims(m, dims));

    SP_CALL(spMeshGetStrides(m, strides));

    param.num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);

//    param.max_num_of_particle = spParticleCapacity(sp);


    param.dims = sizeType2Dim3(dims);

    param.strides = sizeType2Dim3(strides);

    size_type center_min[3], center_max[3];
    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, center_min, center_max, NULL));

    param.center_min = sizeType2Dim3(center_min);

    param.center_max = sizeType2Dim3(center_max);

    Real inv_dx[3], x0[3], dx[3];
    SP_CALL(spMeshGetBox(m, SP_DOMAIN_ALL, x0, NULL));
    SP_CALL(spMeshGetInvDx(m, inv_dx));
    SP_CALL(spMeshGetDx(m, dx));
    param.inv_dx = real2Real3(inv_dx);
    param.dx = real2Real3(dx);
    param.x0 = real2Real3(x0);

    spParallelMemcpyToSymbol(_sp_mesh, &param, sizeof(_spMeshDevice));

    return error_code;
}

//int spFDTDSetupParam(spMesh const *m)
//{
//    _spFDTDParam param;
//    size_type min[3], max[3], strides[3];
//    Real inv_dx[3], x0[3], dx[3];
//
//    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_ALL, min, max, NULL));
//    SP_CALL(spMeshGetStrides(m, strides));
//    SP_CALL(spMeshGetBox(m, SP_DOMAIN_ALL, x0, NULL));
//    SP_CALL(spMeshGetInvDx(m, inv_dx));
//    SP_CALL(spMeshGetDx(m, dx));
//
//    param.min.x = (unsigned int) min[0];
//    param.min.y = (unsigned int) min[1];
//    param.min.z = (unsigned int) min[2];
//
//    param.max.x = (unsigned int) max[0];
//    param.max.y = (unsigned int) max[1];
//    param.max.z = (unsigned int) max[2];
//
//    param.strides.x = (unsigned int) strides[0];
//    param.strides.y = (unsigned int) strides[1];
//    param.strides.z = (unsigned int) strides[2];
//
//    param.inv_dx.x = inv_dx[0];
//    param.inv_dx.y = inv_dx[1];
//    param.inv_dx.z = inv_dx[2];
//
//    param.dx.x = dx[0];
//    param.dx.y = dx[1];
//    param.dx.z = dx[2];
//
//    param.x0.x = x0[0];
//    param.x0.y = x0[1];
//    param.x0.z = x0[2];
//
//
//    spParallelMemcpyToSymbol(_sp_mesh, &param, sizeof(_spFDTDParam));
//
//
//    return SP_SUCCESS;
//}