//
// Created by salmon on 16-8-14.
//

#ifndef SIMPLA_SPFDTDKERNEL_H
#define SIMPLA_SPFDTDKERNEL_H

#include <assert.h>

#include "../sp_lite_def.h"
#include "../spPhysicalConstants.h"
#include "../spMesh.h"
#include "../spField.h"
#include "sp_device.h"


struct spMesh_s;
struct spField_s;

SP_DEVICE_DECLARE_KERNEL (spUpdateFieldFDTDKernel,
                          Real dt,
                          Real3 inv_dx,
                          dim3 min, dim3 max, dim3 strides,
                          Real const *Rho,
                          Real const *Jx,
                          Real const *Jy,
                          Real const *Jz,
                          Real *Ex,
                          Real *Ey,
                          Real *Ez,
                          Real *Bx,
                          Real *By,
                          Real *Bz)
{
//    for (size_type x = min.x + threadIdx.x + blockIdx.x * blockDim.x; x < max.x; x += gridDim.x * blockDim.x)
//        for (size_type y = min.y + threadIdx.y + blockIdx.y * blockDim.y; y < max.y; y += gridDim.y * blockDim.y)
//            for (size_type z = min.z + threadIdx.z + blockIdx.z * blockDim.z; z < max.z; z += gridDim.z * blockDim.z)
    size_type x = min.x + threadIdx.x + blockIdx.x * blockDim.x, y = min.y + threadIdx.y + blockIdx.y * blockDim.y, z =
            min.z + threadIdx.z + blockIdx.z * blockDim.z;
    if (x < max.x && y < max.y && z < max.z)
    {


        size_type s = x * strides.x + y * strides.y + z * strides.z;

        Bx[s] -= ((Ez[s] - Ez[s - strides.y]) * inv_dx.y - (Ey[s] - Ey[s - strides.z]) * inv_dx.z) * 0.5 * dt;
        By[s] -= ((Ex[s] - Ex[s - strides.z]) * inv_dx.z - (Ez[s] - Ez[s - strides.x]) * inv_dx.x) * 0.5 * dt;
        Bz[s] -= ((Ey[s] - Ey[s - strides.x]) * inv_dx.x - (Ex[s] - Ex[s - strides.y]) * inv_dx.y) * 0.5 * dt;

        Ex[s] +=
                ((Bz[s + strides.y] - Bz[s]) * inv_dx.y - (By[s + strides.z] - By[s]) * inv_dx.z) * speed_of_light2
                -
                Jx[s] / epsilon0 * dt;
        Ey[s] +=
                ((Bx[s + strides.z] - Bx[s]) * inv_dx.z - (Bz[s + strides.x] - Bz[s]) * inv_dx.x) * speed_of_light2
                -
                Jy[s] / epsilon0 * dt;
        Ez[s] +=
                ((By[s + strides.x] - By[s]) * inv_dx.x - (Bx[s + strides.y] - Bx[s]) * inv_dx.y) * speed_of_light2
                -
                Jz[s] / epsilon0 * dt;

        Bx[s] -= ((Ez[s] - Ez[s - strides.y]) * inv_dx.y - (Ey[s] - Ey[s - strides.z]) * inv_dx.z) * 0.5 * dt;
        By[s] -= ((Ex[s] - Ex[s - strides.z]) * inv_dx.z - (Ez[s] - Ez[s - strides.x]) * inv_dx.x) * 0.5 * dt;
        Bz[s] -= ((Ey[s] - Ey[s - strides.x]) * inv_dx.x - (Ex[s] - Ex[s - strides.y]) * inv_dx.y) * 0.5 * dt;
    }
}

int spFDTDUpdate(struct spMesh_s const *m, Real dt, const struct spField_s *fRho, const struct spField_s *fJ,
                 struct spField_s *fE, struct spField_s *fB)
{
    if (m == NULL) { return SP_FAILED; }

    assert(spFieldIsSoA(fRho));
    assert(spFieldIsSoA(fJ));
    assert(spFieldIsSoA(fE));
    assert(spFieldIsSoA(fB));

    size_type min[3], max[3], strides[3];
    Real inv_dx[3];
    SP_CALL(spMeshGetInvDx(m, inv_dx));
    SP_CALL(spMeshGetArrayShape(m, SP_DOMAIN_ALL, min, max, strides));

    Real *rho, *J[3], *E[3], *B[3];

    SP_CALL(spFieldSubArray((spField *) fRho, (void **) &rho));

    SP_CALL(spFieldSubArray((spField *) fJ, (void **) J));

    SP_CALL(spFieldSubArray(fE, (void **) E));

    SP_CALL(spFieldSubArray(fB, (void **) B));
    dim3 block_dim = {16, 8, 1};

    dim3 grid_dim = {(max[0] - min[0]) / block_dim.x, (max[1] - min[1]) / block_dim.y, max[2] - min[2]};
    SP_DEVICE_CALL_KERNEL(spUpdateFieldFDTDKernel, grid_dim, block_dim,
                          dt, real2Real3(inv_dx),
                          sizeType2Dim3(min), sizeType2Dim3(max), sizeType2Dim3(strides),
                          (const Real *) rho, (const Real *) J[0], (const Real *) J[1], (const Real *) J[2],
                          E[0], E[1], E[2], B[0], B[1], B[2]);
    spFieldSync(fE);
    spFieldSync(fB);

    return SP_SUCCESS;
}

#endif //SIMPLA_SPFDTDKERNEL_H
