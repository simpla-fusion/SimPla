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


typedef struct
{
    uint3 min;
    uint3 max;
    float3 x0, dx;
    float3 inv_dx;
    uint3 strides;

} _spFDTDParam;

__constant__ _spFDTDParam _fdtd_param;

INLINE __device__ int SPMeshHash(int x, int y, int z)
{
    return
            __mul24((_fdtd_param.max.x - _fdtd_param.min.x + x) % (_fdtd_param.max.x - _fdtd_param.min.x) +
                    _fdtd_param.min.x,
                    _fdtd_param.strides.x) +
            __mul24((_fdtd_param.max.y - _fdtd_param.min.y + y) % (_fdtd_param.max.y - _fdtd_param.min.y) +
                    _fdtd_param.min.y,
                    _fdtd_param.strides.y) +
            __mul24((_fdtd_param.max.z - _fdtd_param.min.z + z) % (_fdtd_param.max.z - _fdtd_param.min.z) +
                    _fdtd_param.min.z,
                    _fdtd_param.strides.z);

}

INLINE __device__  int SPMeshInBox(int x, int y, int z)
{
    return (_fdtd_param.min.x + x < _fdtd_param.max.x && _fdtd_param.min.y + y < _fdtd_param.max.y
            && _fdtd_param.min.z + z < _fdtd_param.max.z);
}

INLINE __device__ void SPMeshPoint(int x, int y, int z, Real *rx, Real *ry, Real *rz)
{
    *rx = _fdtd_param.x0.x + x * _fdtd_param.dx.x;
    *ry = _fdtd_param.x0.y + y * _fdtd_param.dx.y;
    *rz = _fdtd_param.x0.z + z * _fdtd_param.dx.z;

};

int spFDTDSetupParam(spMesh const *m, int tag, size_type *grid_dim, size_type *block_dim)
{
    _spFDTDParam param;
    size_type min[3], max[3], strides[3];
    Real inv_dx[3], x0[3], dx[3];
    SP_CALL(spMeshGetArrayShape(m, tag, min, max, strides));
    SP_CALL(spMeshGetBox(m, tag, x0, NULL));
    SP_CALL(spMeshGetInvDx(m, inv_dx));
    SP_CALL(spMeshGetDx(m, dx));

    param.min.x = (unsigned int) min[0];
    param.min.y = (unsigned int) min[1];
    param.min.z = (unsigned int) min[2];

    param.max.x = (unsigned int) max[0];
    param.max.y = (unsigned int) max[1];
    param.max.z = (unsigned int) max[2];

    param.strides.x = (unsigned int) strides[0];
    param.strides.y = (unsigned int) strides[1];
    param.strides.z = (unsigned int) strides[2];

    param.inv_dx.x = inv_dx[0];
    param.inv_dx.y = inv_dx[1];
    param.inv_dx.z = inv_dx[2];

    param.dx.x = dx[0];
    param.dx.y = dx[1];
    param.dx.z = dx[2];

    param.x0.x = x0[0];
    param.x0.y = x0[1];
    param.x0.z = x0[2];


    SP_CALL(spParallelMemcpyToCache(&_fdtd_param, &param, sizeof(_spFDTDParam)));
    SP_CALL(spParallelThreadBlockDecompose(SP_NUM_OF_THREADS_PER_BLOCK, 3, min, max, grid_dim, block_dim));

    return SP_SUCCESS;
}

SP_DEVICE_DECLARE_KERNEL(spFDTDInitialValueSinKernel, Real *d, Real3 k, Real3 alpha0, Real amp)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (SPMeshInBox(x, y, z))
    {
        Real rx, ry, rz;

        SPMeshPoint(x, y, z, &rx, &ry, &rz);

        d[SPMeshHash(x, y, z)] = (Real) (cos(k.x * rx) * cos(k.y * ry) * cos(k.z * rz)) * amp;
    }
}


#define HALFPI (3.1415926*0.5f)

int spFDTDInitialValueSin(spField *f, Real const *k, Real const *amp)
{

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) f);
    int iform = spMeshAttributeGetForm((spMeshAttribute const *) f);
    int ndims = spMeshGetNDims(m);
    int num_of_sub = spFieldNumberOfSub(f);

    Real *data[num_of_sub];

    size_type dims[3];
    Real x0[3], dx[3];

    SP_CALL(spMeshGetDims(m, dims));
    SP_CALL(spMeshGetOrigin(m, x0));
    SP_CALL(spMeshGetDx(m, dx));


    Real alpha0[9];

    switch (iform)
    {
        case EDGE:
            alpha0[0] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5)));
            alpha0[1] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * x0[1]));
            alpha0[2] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * x0[2]));
            alpha0[3] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * x0[0]));
            alpha0[4] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5)));
            alpha0[5] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * x0[2]));
            alpha0[6] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * x0[0]));
            alpha0[7] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * x0[1]));
            alpha0[8] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5)));
            break;
        case FACE:
            alpha0[0] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * x0[0]));
            alpha0[1] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5)));
            alpha0[2] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5)));
            alpha0[3] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5)));
            alpha0[4] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * x0[1]));
            alpha0[5] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5)));
            alpha0[6] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5)));
            alpha0[7] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5)));
            alpha0[8] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * x0[2]));
            break;
        case VOLUME:

            alpha0[0] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5)));
            alpha0[1] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5)));
            alpha0[2] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5)));
            alpha0[3] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5)));
            alpha0[4] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5)));
            alpha0[5] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5)));
            alpha0[6] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5)));
            alpha0[7] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5)));
            alpha0[8] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5)));
            break;

        case VERTEX:
        default:
            alpha0[0] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * x0[0]));
            alpha0[1] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * x0[1]));
            alpha0[2] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * x0[2]));
            alpha0[3] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * x0[0]));
            alpha0[4] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * x0[1]));
            alpha0[5] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * x0[2]));
            alpha0[6] = (Real) (dims[0] == 1 ? HALFPI : (k[0] * x0[0]));
            alpha0[7] = (Real) (dims[1] == 1 ? HALFPI : (k[1] * x0[1]));
            alpha0[8] = (Real) (dims[2] == 1 ? HALFPI : (k[2] * x0[2]));
    };

    SP_CALL(spFieldSubArray(f, (void **) data));

    size_type grid_dim[3], block_dim[3];

    spFDTDSetupParam(m, SP_DOMAIN_CENTER, grid_dim, block_dim);

    for (int i = 0; i < num_of_sub; ++i)
    {


        SP_DEVICE_CALL_KERNEL(spFDTDInitialValueSinKernel,
                              sizeType2Dim3(grid_dim),
                              sizeType2Dim3(block_dim),
                              data[i],
                              real2Real3(k),
                              real2Real3(alpha0 + i * 3),
                              amp[i]
        );
    }

    SP_CALL(spFieldSync(f));

    return SP_SUCCESS;
};


SP_DEVICE_DECLARE_KERNEL (spUpdateFieldFDTDKernel, Real dt,
                          Real const *Rho, Real const *Jx, Real const *Jy, Real const *Jz,
                          Real *Ex, Real *Ey, Real *Ez,
                          Real *Bx, Real *By, Real *Bz)
{

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (SPMeshInBox(x, y, z))
    {
        int s = SPMeshHash(x, y, z);

//        Bx[s] -=
//                ((Ez[s] - Ez[s - _fdtd_param.strides.y]) * _fdtd_param.inv_dx.y
//                 - (Ey[s] - Ey[s - _fdtd_param.strides.z]) * _fdtd_param.inv_dx.z)
//                * 0.5 * dt;
//        By[s] -=
//                ((Ex[s] - Ex[s - _fdtd_param.strides.z]) * _fdtd_param.inv_dx.z
//                 - (Ez[s] - Ez[s - _fdtd_param.strides.x]) * _fdtd_param.inv_dx.x)
//                * 0.5 * dt;
//        Bz[s] -=
//                ((Ey[s] - Ey[s - _fdtd_param.strides.x]) * _fdtd_param.inv_dx.x
//                 - (Ex[s] - Ex[s - _fdtd_param.strides.y]) * _fdtd_param.inv_dx.y)
//                * 0.5 * dt;


        Bx[s] -=
                ((Ez[s] - Ez[s - _fdtd_param.strides.y]) * _fdtd_param.inv_dx.y
                 - (Ey[s] - Ey[s - _fdtd_param.strides.z]) * _fdtd_param.inv_dx.z)
                * dt;
        By[s] -=
                ((Ex[s] - Ex[s - _fdtd_param.strides.z]) * _fdtd_param.inv_dx.z
                 - (Ez[s] - Ez[s - _fdtd_param.strides.x]) * _fdtd_param.inv_dx.x)
                * dt;
        Bz[s] -=
                ((Ey[s] - Ey[s - _fdtd_param.strides.x]) * _fdtd_param.inv_dx.x
                 - (Ex[s] - Ex[s - _fdtd_param.strides.y]) * _fdtd_param.inv_dx.y)
                * dt;


        Ex[s] +=
                ((Bz[s + _fdtd_param.strides.y] - Bz[s]) * _fdtd_param.inv_dx.y
                 - (By[s + _fdtd_param.strides.z] - By[s]) * _fdtd_param.inv_dx.z)
                * speed_of_light2 * dt - Jx[s] / epsilon0 * dt;
        Ey[s] +=
                ((Bx[s + _fdtd_param.strides.z] - Bx[s]) * _fdtd_param.inv_dx.z
                 - (Bz[s + _fdtd_param.strides.x] - Bz[s]) * _fdtd_param.inv_dx.x)
                * speed_of_light2 * dt - Jy[s] / epsilon0 * dt;
        Ez[s] +=
                ((By[s + _fdtd_param.strides.x] - By[s]) * _fdtd_param.inv_dx.x
                 - (Bx[s + _fdtd_param.strides.y] - Bx[s]) * _fdtd_param.inv_dx.y)
                * speed_of_light2 * dt - Jz[s] / epsilon0 * dt;


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

    spFDTDSetupParam(spMeshAttributeGetMesh((spMeshAttribute *) fE), SP_DOMAIN_AFFECT_1, grid_dim, block_dim);

    SP_DEVICE_CALL_KERNEL(spUpdateFieldFDTDKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          dt, (const Real *) rho, (const Real *) J[0], (const Real *) J[1], (const Real *) J[2],
                          E[0], E[1], E[2], B[0], B[1], B[2]);
    spFieldSync(fE);
    spFieldSync(fB);

    return SP_SUCCESS;
}


#endif //SIMPLA_SPFDTD_IMPL_H_H
