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
#include "../spPhysicalConstants.h"
#include "spMesh.cu.h"
#include "sp_device.h"

//
//typedef struct
//{
//    uint3 min;
//    uint3 max;
//    Real3 x0, dx;
//    Real3 inv_dx;
//    uint3 strides;
//
//} _spFDTDParam;

//__constant__ _spFDTDParam _sp_mesh;
//
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

SP_DEVICE_DECLARE_KERNEL(spFDTDInitialValueSinKernel, Real *d, Real3 k, Real3 alpha0, Real amp)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (SPMeshInCenter(x, y, z))
    {
        Real rx = alpha0.x, ry = alpha0.y, rz = alpha0.z;
        SPMeshPoint(x, y, z, &rx, &ry, &rz);
        int s = SPMeshHash(x, y, z);
        d[s] = (Real) (cos(k.x * rx) * cos(k.y * ry) * cos(k.z * rz)) * amp;
    }
}

int spFDTDInitialValueSin(spField *f, Real const *k, Real const *amp)
{

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) f);
    int iform = spMeshAttributeGetForm((spMeshAttribute const *) f);
    int ndims = spMeshGetNDims(m);
    int num_of_sub = spFieldNumberOfSub(f);

    Real *data[num_of_sub];


    SP_CALL(spFieldSubArray(f, (void **) data));

    size_type grid_dim[3], block_dim[3];
    SP_CALL(spMeshGetDims(m, grid_dim));
    SP_CALL(spParallelThreadBlockDecompose(256, grid_dim, block_dim));
    Real alpha0[4][9] =
            {
                    {/**/ 0,   0,   0,   /**/ 0,   0,   0,   /**/  0,   0,   0   /**/},
                    {/**/ 0.5, 0,   0,   /**/ 0,   0.5, 0,   /**/  0,   0,   0.5 /**/},
                    {/**/ 0,   0.5, 0.5, /**/ 0.5, 0,   0.5, /**/  0.5, 0.5, 0   /**/},
                    {/**/ 0.5, 0.5, 0.5, /**/ 0.5, 0.5, 0.5, /**/  0.5, 0.5, 0.5 /**/},
            };


    for (int i = 0; i < num_of_sub; ++i)
    {


        SP_CALL_DEVICE_KERNEL(spFDTDInitialValueSinKernel,
                              sizeType2Dim3(grid_dim),
                              sizeType2Dim3(block_dim),
                              data[i],
                              real2Real3(k),
                              real2Real3(&alpha0[iform][i * 3]),
                              amp[i]
        );
    }

    SP_CALL(spFieldSync(f));
    return SP_SUCCESS;
};


SP_DEVICE_DECLARE_KERNEL (spUpdateFieldFDTDKernelPushE, Real dt,
                          Real *Ex, Real *Ey, Real *Ez,
                          Real *Bx, Real *By, Real *Bz,
                          Real const *Rho, Real const *Jx, Real const *Jy, Real const *Jz)
{

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (SPMeshInCenter(x, y, z))
    {
        int s = SPMeshHash(x, y, z);

        Ex[s] += -Jx[s] / epsilon0 * dt
                 + ((Bz[s] - Bz[s - _sp_mesh.strides.y]) * _sp_mesh.inv_dx.y
                    - (By[s] - By[s - _sp_mesh.strides.z]) * _sp_mesh.inv_dx.z)
                   * speed_of_light2 * dt;
        Ey[s] += -Jy[s] / epsilon0 * dt
                 + ((Bx[s] - Bx[s - _sp_mesh.strides.z]) * _sp_mesh.inv_dx.z
                    - (Bz[s] - Bz[s - _sp_mesh.strides.x]) * _sp_mesh.inv_dx.x)
                   * speed_of_light2 * dt;
        Ez[s] += -Jz[s] / epsilon0 * dt
                 + ((By[s] - By[s - _sp_mesh.strides.x]) * _sp_mesh.inv_dx.x
                    - (Bx[s] - Bx[s - _sp_mesh.strides.y]) * _sp_mesh.inv_dx.y)
                   * speed_of_light2 * dt;
    }
}

SP_DEVICE_DECLARE_KERNEL (spUpdateFieldFDTDKernelPushB, Real dt,
                          Real *Ex, Real *Ey, Real *Ez,
                          Real *Bx, Real *By, Real *Bz)
{

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (SPMeshInBox(x, y, z))
    {
        int s = SPMeshHash(x, y, z);

        Bx[s] -= ((Ez[s + _sp_mesh.strides.y] - Ez[s]) * _sp_mesh.inv_dx.y
                  - (Ey[s + _sp_mesh.strides.z] - Ey[s]) * _sp_mesh.inv_dx.z) * dt;
        By[s] -= ((Ex[s + _sp_mesh.strides.z] - Ex[s]) * _sp_mesh.inv_dx.z
                  - (Ez[s + _sp_mesh.strides.x] - Ez[s]) * _sp_mesh.inv_dx.x) * dt;
        Bz[s] -= ((Ey[s + _sp_mesh.strides.x] - Ey[s]) * _sp_mesh.inv_dx.x
                  - (Ex[s + _sp_mesh.strides.y] - Ex[s]) * _sp_mesh.inv_dx.y) * dt;

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

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) fE);

//    spFDTDSetupParam(m);

    size_type grid_dim[3], block_dim[3];

    SP_CALL(spMeshGetDims(m, grid_dim));
    SP_CALL(spParallelThreadBlockDecompose(NUMBER_OF_THREADS_PER_BLOCK, grid_dim, block_dim));

    SP_CALL_DEVICE_KERNEL(spUpdateFieldFDTDKernelPushB, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim), 0.5 * dt,
                          E[0], E[1], E[2], B[0], B[1], B[2]);

    SP_CALL(spFieldSync(fB));

    SP_CALL_DEVICE_KERNEL(spUpdateFieldFDTDKernelPushE, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim), dt,
                          E[0], E[1], E[2], B[0], B[1], B[2],
                          (const Real *) rho, (const Real *) J[0], (const Real *) J[1], (const Real *) J[2]);

    SP_CALL(spFieldSync(fE));


    SP_CALL_DEVICE_KERNEL(spUpdateFieldFDTDKernelPushB, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim), 0.5 * dt,
                          E[0], E[1], E[2], B[0], B[1], B[2]);

    SP_CALL(spFieldSync(fB));


    return SP_SUCCESS;
}

SP_DEVICE_DECLARE_KERNEL (spFDTDDivKernel, Real const *Jx, Real const *Jy, Real const *Jz, Real *rho)
{

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (SPMeshInCenter(x, y, z))
    {
        int s = SPMeshHash(x, y, z);
        rho[s] +=
                (Jx[s + _sp_mesh.strides.x] - Jx[s]) * _sp_mesh.inv_dx.x +
                (Jy[s + _sp_mesh.strides.y] - Jy[s]) * _sp_mesh.inv_dx.y +
                (Jz[s + _sp_mesh.strides.z] - Jz[s]) * _sp_mesh.inv_dx.z;

    }
}

int spFDTDDiv(const spField *fJ, spField *fRho)
{

    assert(spFieldIsSoA(fRho));
    assert(spFieldIsSoA(fJ));

    Real *rho, *J[3];
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) fRho);

    size_type grid_dim[3], block_dim[3];

    SP_CALL(spMeshGetGlobalDims(m, grid_dim));

    SP_CALL(spParallelThreadBlockDecompose(128, grid_dim, block_dim));

    SP_CALL(spFieldSubArray((spField *) fRho, (void **) &rho));

    SP_CALL(spFieldSubArray((spField *) fJ, (void **) J));


//    SP_CALL(spFDTDSetupParam(m));

    SP_CALL_DEVICE_KERNEL(spFDTDDivKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (const Real *) J[0], (const Real *) J[1], (const Real *) J[2], rho);

    SP_CALL(spFieldSync(fRho));


    return SP_SUCCESS;
}

SP_DEVICE_DECLARE_KERNEL (spFDTDMultiplyByScalarKernel, Real *rho, Real a)
{

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (SPMeshInCenter(x, y, z)) { rho[SPMeshHash(x, y, z)] *= a; }
}

int spFDTDMultiplyByScalar(spField *fRho, Real a)
{

    assert(spFieldIsSoA(fRho));

    Real *rho;

    SP_CALL(spFieldSubArray((spField *) fRho, (void **) &rho));

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) fRho);

    size_type grid_dim[3], block_dim[3];

//    SP_CALL(spFDTDSetupParam(m));

    SP_CALL_DEVICE_KERNEL(spFDTDMultiplyByScalarKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim), rho, a);

    SP_CALL(spFieldSync(fRho));


    return SP_SUCCESS;
}


#endif //SIMPLA_SPFDTD_IMPL_H_H
