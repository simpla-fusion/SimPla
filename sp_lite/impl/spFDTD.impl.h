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
#include "spParallel.impl.h"


struct spMesh_s;
struct spField_s;

SP_DEVICE_DECLARE_KERNEL (spUpdateFieldFDTDKernel, Real dt, Real3 dt_inv, dim3 N, dim3 I, Real const *Rho, Real const *Jx,
                          Real const *Jy, Real const *Jz, Real *Ex, Real *Ey, Real *Ez, Real *Bx,
                          Real *By, Real *Bz)
{
    size_type x = (threadIdx.x + blockIdx.x * blockDim.x + N.x) % N.x;
    size_type y = (threadIdx.y + blockIdx.y * blockDim.y + N.y) % N.y;
    size_type z = (threadIdx.z + blockIdx.z * blockDim.z + N.z) % N.z;

    size_type s = x * I.x + y * I.y + z * I.z;
    Bx[s] -= ((Ez[s] - Ez[s - I.y]) * dt_inv.y - (Ey[s] - Ey[s - I.z]) * dt_inv.z) * 0.5;
    By[s] -= ((Ex[s] - Ex[s - I.z]) * dt_inv.z - (Ez[s] - Ez[s - I.x]) * dt_inv.x) * 0.5;
    Bz[s] -= ((Ey[s] - Ey[s - I.x]) * dt_inv.x - (Ex[s] - Ex[s - I.y]) * dt_inv.y) * 0.5;

    Ex[s] += ((Bz[s + I.y] - Bz[s]) * dt_inv.y - (By[s + I.z] - By[s]) * dt_inv.z) * speed_of_light2 -
             Jx[s] / epsilon0 * dt;
    Ey[s] += ((Bx[s + I.z] - Bx[s]) * dt_inv.z - (Bz[s + I.x] - Bz[s]) * dt_inv.x) * speed_of_light2 -
             Jy[s] / epsilon0 * dt;
    Ez[s] += ((By[s + I.x] - By[s]) * dt_inv.x - (Bx[s + I.y] - Bx[s]) * dt_inv.y) * speed_of_light2 -
             Jz[s] / epsilon0 * dt;

    Bx[s] -= ((Ez[s] - Ez[s - I.y]) * dt_inv.y - (Ey[s] - Ey[s - I.z]) * dt_inv.z) * 0.5;
    By[s] -= ((Ex[s] - Ex[s - I.z]) * dt_inv.z - (Ez[s] - Ez[s - I.x]) * dt_inv.x) * 0.5;
    Bz[s] -= ((Ey[s] - Ey[s - I.x]) * dt_inv.x - (Ex[s] - Ex[s - I.y]) * dt_inv.y) * 0.5;
}


int spFDTDUpdate(struct spMesh_s const *m, Real dt, const struct spField_s *fRho, const struct spField_s *fJ,
                 struct spField_s *fE, struct spField_s *fB)
{
    if (m == NULL) { return SP_FAILED; }

    assert(spFieldIsSoA(fRho));
    assert(spFieldIsSoA(fJ));
    assert(spFieldIsSoA(fE));
    assert(spFieldIsSoA(fB));

    dim3 block_dim, thread_dim;

    size_type dims[4], start[4], count[4];
    size_type strides[3];
    Real inv_dx[3];
    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_ALL, dims, start, count));
    SP_CALL(spMeshGetStrides(m, strides));
    SP_CALL(spMeshGetInvDx(m, inv_dx));

    Real dt_inv[3] = {dt * inv_dx[0], dt * inv_dx[1], dt * inv_dx[2]};

    Real *rho, *J[3], *E[3], *B[3];

    SP_CALL(spFieldSubArray((spField *) fRho, (void **) &rho));

    SP_CALL(spFieldSubArray((spField *) fJ, (void **) J));

    SP_CALL(spFieldSubArray(fE, (void **) E));

    SP_CALL(spFieldSubArray(fB, (void **) B));

    dim3 threads = {1, 0, 0};

    SP_DEVICE_CALL_KERNEL(spUpdateFieldFDTDKernel, sizeType2Dim3(dims), threads,
                          dt, real2Real3(dt_inv), sizeType2Dim3(dims),
                          sizeType2Dim3(strides), (const Real *) rho, (const Real *) J[0], (const Real *) J[1],
                          (const Real *) J[2], E[0], E[1], E[2], B[0], B[1], B[2]);

    spFieldSync(fE);
    spFieldSync(fB);

    return SP_SUCCESS;
}

#endif //SIMPLA_SPFDTDKERNEL_H