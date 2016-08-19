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
#include "spContext.impl.h"

struct spMesh_s;
struct spField_s;

SP_DEVICE_DECLARE_KERNEL (spUpdateFieldFDTDKernel, Real dt,
                          Real const *Rho, Real const *Jx, Real const *Jy, Real const *Jz,
                          Real *Ex, Real *Ey, Real *Ez,
                          Real *Bx, Real *By, Real *Bz)
{
//    for (size_type x = min.x + threadIdx.x + blockIdx.x * blockDim.x; x < max.x; x += gridDim.x * blockDim.x)
//        for (size_type y = min.y + threadIdx.y + blockIdx.y * blockDim.y; y < max.y; y += gridDim.y * blockDim.y)
//            for (size_type z = min.z + threadIdx.z + blockIdx.z * blockDim.z; z < max.z; z += gridDim.z * blockDim.z)
    size_type x = sp_ctx_d.min.x + threadIdx.x + blockIdx.x * blockDim.x;
    size_type y = sp_ctx_d.min.y + threadIdx.y + blockIdx.y * blockDim.y;
    size_type z = sp_ctx_d.min.z + threadIdx.z + blockIdx.z * blockDim.z;
    if (x < sp_ctx_d.max.x && y < sp_ctx_d.max.y && z < sp_ctx_d.max.z)
    {


        size_type s = x * sp_ctx_d.strides.x + y * sp_ctx_d.strides.y + z * sp_ctx_d.strides.z;

        Bx[s] -=
            ((Ez[s] - Ez[s - sp_ctx_d.strides.y]) * sp_ctx_d.inv_dx.y
                - (Ey[s] - Ey[s - sp_ctx_d.strides.z]) * sp_ctx_d.inv_dx.z)
                * 0.5 * dt;
        By[s] -=
            ((Ex[s] - Ex[s - sp_ctx_d.strides.z]) * sp_ctx_d.inv_dx.z
                - (Ez[s] - Ez[s - sp_ctx_d.strides.x]) * sp_ctx_d.inv_dx.x)
                * 0.5 * dt;
        Bz[s] -=
            ((Ey[s] - Ey[s - sp_ctx_d.strides.x]) * sp_ctx_d.inv_dx.x
                - (Ex[s] - Ex[s - sp_ctx_d.strides.y]) * sp_ctx_d.inv_dx.y)
                * 0.5 * dt;

        Ex[s] +=
            ((Bz[s + sp_ctx_d.strides.y] - Bz[s]) * sp_ctx_d.inv_dx.y
                - (By[s + sp_ctx_d.strides.z] - By[s]) * sp_ctx_d.inv_dx.z)
                * speed_of_light2 - Jx[s] / epsilon0 * dt;
        Ey[s] +=
            ((Bx[s + sp_ctx_d.strides.z] - Bx[s]) * sp_ctx_d.inv_dx.z
                - (Bz[s + sp_ctx_d.strides.x] - Bz[s]) * sp_ctx_d.inv_dx.x)
                * speed_of_light2 - Jy[s] / epsilon0 * dt;
        Ez[s] +=
            ((By[s + sp_ctx_d.strides.x] - By[s]) * sp_ctx_d.inv_dx.x
                - (Bx[s + sp_ctx_d.strides.y] - Bx[s]) * sp_ctx_d.inv_dx.y)
                * speed_of_light2 - Jz[s] / epsilon0 * dt;

        Bx[s] -=
            ((Ez[s] - Ez[s - sp_ctx_d.strides.y]) * sp_ctx_d.inv_dx.y
                - (Ey[s] - Ey[s - sp_ctx_d.strides.z]) * sp_ctx_d.inv_dx.z)
                * 0.5 * dt;
        By[s] -=
            ((Ex[s] - Ex[s - sp_ctx_d.strides.z]) * sp_ctx_d.inv_dx.z
                - (Ez[s] - Ez[s - sp_ctx_d.strides.x]) * sp_ctx_d.inv_dx.x)
                * 0.5 * dt;
        Bz[s] -=
            ((Ey[s] - Ey[s - sp_ctx_d.strides.x]) * sp_ctx_d.inv_dx.x
                - (Ex[s] - Ex[s - sp_ctx_d.strides.y]) * sp_ctx_d.inv_dx.y)
                * 0.5 * dt;
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

    Real *rho, *J[3], *E[3], *B[3];

    SP_CALL(spFieldSubArray((spField *) fRho, (void **) &rho));

    SP_CALL(spFieldSubArray((spField *) fJ, (void **) J));

    SP_CALL(spFieldSubArray(fE, (void **) E));

    SP_CALL(spFieldSubArray(fB, (void **) B));

    SP_DEVICE_CALL_KERNEL(spUpdateFieldFDTDKernel, sp_ctx.grid_dim, sp_ctx.block_dim, dt,
                          (const Real *) rho, (const Real *) J[0], (const Real *) J[1], (const Real *) J[2],
                          E[0], E[1], E[2], B[0], B[1], B[2]);
    spFieldSync(fE);
    spFieldSync(fB);

    return SP_SUCCESS;
}

#endif //SIMPLA_SPFDTDKERNEL_H
