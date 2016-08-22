//
// Created by salmon on 16-8-22.
//

#ifndef SIMPLA_SPFDTD_IMPL_H_H
#define SIMPLA_SPFDTD_IMPL_H_H


#include "../sp_lite_def.h"

#include <math.h>
#include <assert.h>

#include "../spMesh.h"
#include "../spField.h"
#include "../spPICBoris.h"
#include "../spRandom.h"
#include "../spPhysicalConstants.h"

#include "sp_device.h"

#ifdef NUM_OF_THREADS_PER_BLOCK
#   define SP_NUM_OF_THREADS_PER_BLOCK NUM_OF_THREADS_PER_BLOCK
#else
#   define SP_NUM_OF_THREADS_PER_BLOCK 128
#endif


typedef struct
{
    uint3 min;
    uint3 max;
    float3 inv_dx;
    uint3 strides;

} _spFDTDParam;

__constant__ _spFDTDParam _fdtd_param;

int spFDTDSetupParam(spMesh const *m, int tag, size_type *grid_dim, size_type *block_dim)
{
    _spFDTDParam param;
    size_type min[3], max[3], strides[3];
    Real inv_dx[3];
    SP_CALL(spMeshGetArrayShape(m, tag, min, max, strides));
    SP_CALL(spMeshGetInvDx(m, inv_dx));
    param.min = sizeType2Dim3(min);
    param.max = sizeType2Dim3(max);
    param.strides = sizeType2Dim3(strides);
    param.inv_dx = real2Real3(inv_dx);
    SP_CALL(spParallelMemcpyToCache(&_pic_param, &param, sizeof(_spFDTDParam)));
    SP_CALL(spMeshThreadBlockDecompose(m, SP_NUM_OF_THREADS_PER_BLOCK, grid_dim, block_dim));

}

SP_DEVICE_DECLARE_KERNEL (spUpdateFieldFDTDKernel, Real dt,
                          Real const *Rho, Real const *Jx, Real const *Jy, Real const *Jz,
                          Real *Ex, Real *Ey, Real *Ez,
                          Real *Bx, Real *By, Real *Bz)
{
//    for (size_type x = min.x + threadIdx.x + blockIdx.x * blockDim.x; x < max.x; x += gridDim.x * blockDim.x)
//        for (size_type y = min.y + threadIdx.y + blockIdx.y * blockDim.y; y < max.y; y += gridDim.y * blockDim.y)
//            for (size_type z = min.z + threadIdx.z + blockIdx.z * blockDim.z; z < max.z; z += gridDim.z * blockDim.z)
    size_type x = _fdtd_param.min.x + threadIdx.x + blockIdx.x * blockDim.x;
    size_type y = _fdtd_param.min.y + threadIdx.y + blockIdx.y * blockDim.y;
    size_type z = _fdtd_param.min.z + threadIdx.z + blockIdx.z * blockDim.z;
    if (x < _fdtd_param.max.x && y < _fdtd_param.max.y && z < _fdtd_param.max.z)
    {


        size_type s = x * _fdtd_param.strides.x + y * _fdtd_param.strides.y + z * _fdtd_param.strides.z;

        Bx[s] -=
            ((Ez[s] - Ez[s - _fdtd_param.strides.y]) * _fdtd_param.inv_dx.y
                - (Ey[s] - Ey[s - _fdtd_param.strides.z]) * _fdtd_param.inv_dx.z)
                * 0.5 * dt;
        By[s] -=
            ((Ex[s] - Ex[s - _fdtd_param.strides.z]) * _fdtd_param.inv_dx.z
                - (Ez[s] - Ez[s - _fdtd_param.strides.x]) * _fdtd_param.inv_dx.x)
                * 0.5 * dt;
        Bz[s] -=
            ((Ey[s] - Ey[s - _fdtd_param.strides.x]) * _fdtd_param.inv_dx.x
                - (Ex[s] - Ex[s - _fdtd_param.strides.y]) * _fdtd_param.inv_dx.y)
                * 0.5 * dt;

        Ex[s] +=
            ((Bz[s + _fdtd_param.strides.y] - Bz[s]) * _fdtd_param.inv_dx.y
                - (By[s + _fdtd_param.strides.z] - By[s]) * _fdtd_param.inv_dx.z)
                * speed_of_light2 - Jx[s] / epsilon0 * dt;
        Ey[s] +=
            ((Bx[s + _fdtd_param.strides.z] - Bx[s]) * _fdtd_param.inv_dx.z
                - (Bz[s + _fdtd_param.strides.x] - Bz[s]) * _fdtd_param.inv_dx.x)
                * speed_of_light2 - Jy[s] / epsilon0 * dt;
        Ez[s] +=
            ((By[s + _fdtd_param.strides.x] - By[s]) * _fdtd_param.inv_dx.x
                - (Bx[s + _fdtd_param.strides.y] - Bx[s]) * _fdtd_param.inv_dx.y)
                * speed_of_light2 - Jz[s] / epsilon0 * dt;

        Bx[s] -=
            ((Ez[s] - Ez[s - _fdtd_param.strides.y]) * _fdtd_param.inv_dx.y
                - (Ey[s] - Ey[s - _fdtd_param.strides.z]) * _fdtd_param.inv_dx.z)
                * 0.5 * dt;
        By[s] -=
            ((Ex[s] - Ex[s - _fdtd_param.strides.z]) * _fdtd_param.inv_dx.z
                - (Ez[s] - Ez[s - _fdtd_param.strides.x]) * _fdtd_param.inv_dx.x)
                * 0.5 * dt;
        Bz[s] -=
            ((Ey[s] - Ey[s - _fdtd_param.strides.x]) * _fdtd_param.inv_dx.x
                - (Ex[s] - Ex[s - _fdtd_param.strides.y]) * _fdtd_param.inv_dx.y)
                * 0.5 * dt;
    }
}

int spFDTDUpdate(Real dt, const spField *fRho, const spField *fJ, spField *fE, spField *fB)
{
    assert(spFieldIsSoA(fRho));
    assert(spFieldIsSoA(fJ));
    assert(spFieldIsSoA(fE));
    assert(spFieldIsSoA(fB));

    Real *rho, *J[3], *E[3], *B[3];

    SP_CALL(spFieldSubArray((spField *) fRho, (void **) &rho));

    SP_CALL(spFieldSubArray((spField *) fJ, (void **) J));

    SP_CALL(spFieldSubArray(fE, (void **) E));

    SP_CALL(spFieldSubArray(fB, (void **) B));


    size_type grid_dim[3], block_dim[3];

    spFDTDSetupParam(spMeshAttributeGetMesh((spMeshAttribute *) fE), SP_DOMAIN_ALL, grid_dim, block_dim);

    SP_DEVICE_CALL_KERNEL(spUpdateFieldFDTDKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          dt, (const Real *) rho, (const Real *) J[0], (const Real *) J[1], (const Real *) J[2],
                          E[0], E[1], E[2], B[0], B[1], B[2]);
    spFieldSync(fE);
    spFieldSync(fB);

    return SP_SUCCESS;
}


#endif //SIMPLA_SPFDTD_IMPL_H_H
