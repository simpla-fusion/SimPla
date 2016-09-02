//
// Created by salmon on 16-8-4.
//

#ifndef SIMPLA_SPBORIS_DEVICE_H
#define SIMPLA_SPBORIS_DEVICE_H

extern "C" {
#include "../../sp_lite_def.h"

#include <math.h>
#include <assert.h>

#include "../../spMesh.h"
#include "../../spField.h"
#include "../../spPICBoris.h"
#include "../../spRandom.h"
#include "../../spPhysicalConstants.h"

#include "../sp_device.h"
#include "../spBoris.impl.h"
};

//__constant__ _spPICBorisParam _pic_param;


INLINE __device__ int _spMeshHash(int x, int y, int z)
{
    return __mul24(x, _pic_param.strides.x) +
           __mul24(y, _pic_param.strides.y) +
           __mul24(z, _pic_param.strides.z);
}

INLINE __device__  int _SPMeshInBox(int x, int y, int z)
{
    return (_pic_param.min.x + x < _pic_param.max.x && _pic_param.min.y + y < _pic_param.max.y
            && _pic_param.min.z + z < _pic_param.max.z);
}

INLINE  __device__ void
spParticleInitializeBoris(boris_particle *sp, size_type s, Real vT, Real f0)
{

    sp->f[s] = f0;
    sp->w[s] = 0.0;

    sp->vx[s] *= vT;
    sp->vy[s] *= vT;
    sp->vz[s] *= vT;
}

SP_DEVICE_DECLARE_KERNEL(spParticleInitializeBorisYeeKernel, boris_particle *sp, Real vT, Real f0, int PIC)
{
    size_type threadId = threadIdx.x * blockDim.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.z;
    size_type x = _pic_param.min.x + blockIdx.x;
    size_type y = _pic_param.min.y + blockIdx.y;
    size_type z = _pic_param.min.z + blockIdx.z;


    if (threadId < PIC)
    {
        size_type s0 = x * _pic_param.strides.x + y * _pic_param.strides.y + z * _pic_param.strides.z;

//#pragma unroll
        for (size_type s = s0 * _pic_param.max_pic + threadIdx.x, se = s + PIC; s < se;
             s += blockDim.x)
        {
            sp->id[s] = 0;
            sp->f[s] = f0;
            sp->w[s] = 0.0;

            sp->vx[s] *= vT;
            sp->vy[s] *= vT;
            sp->vz[s] *= vT;
//            spParticleInitializeBoris(sp, s, vT, f0);
        }
    }

}

int spParticleInitializeBorisYee(spParticle *sp, Real n0, Real T0)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    SP_CALL(spParticleDeploy(sp));

    int dist_type[6] = {SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_UNIFORM,
                        SP_RAND_NORMAL, SP_RAND_NORMAL, SP_RAND_NORMAL};

    SP_CALL(spParticleInitialize(sp, dist_type));

    Real dx[3];

    SP_CALL(spMeshGetDx(m, dx));

    Real vT = (Real) sqrt(2.0 * SI_Boltzmann_constant * T0 / spParticleGetMass(sp));

    Real f0 = n0 * dx[0] * dx[1] * dx[2] / spParticleGetPIC(sp) * spParticleGetCharge(sp);

    void **device_data;

    SP_CALL(spParticleGetAllAttributeData_device(sp, &device_data));

    size_type grid_dim[3], block_dim[3];

    SP_CALL(spPICBorisSetupParam(sp, SP_DOMAIN_CENTER, grid_dim, block_dim));

    SP_DEVICE_CALL_KERNEL(spParticleInitializeBorisYeeKernel,
                          sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (boris_particle *) device_data, vT, f0, spParticleGetPIC(sp));

    SP_CALL(spParticleSync(sp));

    return SP_SUCCESS;
}

/******************************************************************************************/


SP_DEVICE_DECLARE_KERNEL (spParticleUpdateBorisYeeKernel, Real dt,
                          boris_particle *sp,
                          Real const *Ex, Real const *Ey, Real const *Ez,
                          Real const *Bx, Real const *By, Real const *Bz)
{
    int threadId = threadIdx.x * blockDim.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.z;
    int num_of_thread = blockDim.x * blockDim.x * blockDim.x;

    int x = _pic_param.min.x + blockIdx.x;
    int y = _pic_param.min.y + blockIdx.y;
    int z = _pic_param.min.z + blockIdx.z;
    int s0 = _spMeshHash(_pic_param.min.x + blockIdx.x, _pic_param.min.y + blockIdx.y, _pic_param.min.z + blockIdx.z);

//    __shared__  Real cE[27 * 3];
//    __shared__  Real cB[27 * 3];
//    __shared__ int dest_tail;

    __shared__  Real cE[6];
    __shared__  Real cB[6];


    spParallelSyncThreads();
    if (threadId == 0)
    {

        cE[0] = Ex[s0 - _pic_param.strides.x];
        cE[1] = Ex[s0 /*                  */];
        cE[2] = Ey[s0 - _pic_param.strides.y];
        cE[3] = Ey[s0 /*                  */];
        cE[4] = Ez[s0 - _pic_param.strides.z];
        cE[5] = Ez[s0 /*                  */];


        cB[0] = Bx[s0 - _pic_param.strides.x];
        cB[1] = Bx[s0 /*                  */];
        cB[2] = By[s0 - _pic_param.strides.y];
        cB[3] = By[s0 /*                  */];
        cB[4] = Bz[s0 - _pic_param.strides.z];
        cB[5] = Bz[s0 /*                  */];


    }
    spParallelSyncThreads();

    //    if (num_of_thread < 27 * 3 && threadId == 0)
//    {
//
//
//        for (int i = -1; i <= 1; ++i)
//            for (int j = -1; j <= 1; ++j)
//                for (int k = -1; k <= 1; ++k)
//                {
//                    int s1 = 1 + i + (1 + j) * 3 + (1 + k) * 9;
//                    int s2 = _spMeshHash(x + i, y + j, z + k);
//
//                    cE[s1] = Ex[s2];
//                    cB[s1] = Bx[s2];
//                    cE[s1 + 27] = Ey[s2];
//                    cB[s1 + 27] = By[s2];
//                    cE[s1 + 27 * 2] = Ez[s2];
//                    cB[s1 + 27 * 2] = Bz[s2];
//
//                }
//
//    } else if (threadId < 27 * 3)
//    {
//
//        int s2 = _spMeshHash(x + ((threadId % 3) - 1), y + ((threadId / 3) % 3 - 1), z + ((threadId / 9) - 1));
//
//        cE[threadId] = Ex[s2];
//        cB[threadId] = Bx[s2];
//        cE[threadId + 27] = Ey[s2];
//        cB[threadId + 27] = By[s2];
//        cE[threadId + 54] = Ez[s2];
//        cB[threadId + 54] = Bz[s2];
//    }
    int dest0 = s0 * _pic_param.max_pic;
    /**
     *   -1 -> 0b11
     *   00 -> 0b00
     *    1 -> 0b01
     *    (i+4)& 0x3
     */
    __shared__ int src[256], dest[256];
    __shared__ int src_tail, dest_tail;
    int tag;

    for (int i = -1; i < 1; ++i)
        for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k)
            {
                int src0 = _spMeshHash(x + i, y + j, z + k) * _pic_param.max_pic;

                while (1)
                {
                    spParallelSyncThreads();

                    while (src[threadId] < src0 + _pic_param.max_pic &&
                           sp->id[src[threadId]] != tag) { src[threadId] = atomicAddInt(&src_tail, 1); }

                    if (i != 0 || j != 0 || k != 0)
                    {
                        while (dest[threadId] < dest0 + _pic_param.max_pic &&
                               sp->id[dest[threadId]] != 0x26) { dest[threadId] = atomicAddInt(&dest_tail, 1); }
                    } else
                    {
                        dest[threadId] = src[threadId];
                    }
                    spParallelSyncThreads();

                    if (dest[threadId] >= dest0 + _pic_param.max_pic ||
                        src[threadId] >= src0 + _pic_param.max_pic) { break; }

                    struct boris_particle_p_s p;

                    spParticlePopBoris(sp, src[threadId], &p);

                    Real E[3], B[3];

                    E[0] = cE[0] * (0.5f - p.rx) + cE[1] * (0.5f + p.rx);
                    E[1] = cE[2] * (0.5f - p.ry) + cE[3] * (0.5f + p.ry);
                    E[2] = cE[4] * (0.5f - p.rz) + cE[5] * (0.5f + p.rz);

                    spParticleMoveBoris(dt, &p, (Real const *) E, (Real const *) B);

                    p.rx += 4.5;
                    p.ry += 4.5;
                    p.rz += 4.5;
                    p.id = p.id & (~0x3F)
                           | (((int) (p.rx) & 0x3) << 0)
                           | (((int) (p.ry) & 0x3) << 2)
                           | (((int) (p.rz) & 0x3) << 4);

                    p.rx -= (int) (p.rx) + .5;
                    p.ry -= (int) (p.ry) + .5;
                    p.rz -= (int) (p.rz) + .5;

                    spParticlePushBoris(sp, dest[threadId], &p);

                    src[threadId] += num_of_thread;
                    dest[threadId] += num_of_thread;
                }
            }
};

SP_DEVICE_DECLARE_KERNEL (spParticleAccumlateBorisYeeKernel,
                          boris_particle *sp, Real *fJx, Real *fJy, Real *fJz, Real *fRho)
{
    int threadId = threadIdx.x * blockDim.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.z;
    int num_of_thread = blockDim.x * blockDim.x * blockDim.x;

    int x = _pic_param.min.x + blockIdx.x;
    int y = _pic_param.min.y + blockIdx.y;
    int z = _pic_param.min.z + blockIdx.z;

    int s0 = _spMeshHash(x, y, z);

    Real J[6], rho = 0;

    for (int i = 0; i < 6; ++i) { J[i] = 0; }

    for (int s = s0 * _pic_param.max_pic + threadId, se = s + _pic_param.max_pic; s < se; s += num_of_thread)
    {

        if (sp->id[s] == 0)
        {

            Real f = sp->f[s];
            J[0] += (sp->rx[s] >= 0.0) ? 0 : sp->vx[s] * _pic_param.invD.x * f;
            J[1] += (sp->rx[s] < 0.0) ? 0 : -sp->vx[s] * _pic_param.invD.x * f;
            J[2] += (sp->ry[s] >= 0.0) ? 0 : sp->vy[s] * _pic_param.invD.y * f;
            J[3] += (sp->ry[s] < 0.0) ? 0 : -sp->vy[s] * _pic_param.invD.y * f;
            J[4] += (sp->rz[s] >= 0.0) ? 0 : sp->vz[s] * _pic_param.invD.z * f;
            J[5] += (sp->rz[s] < 0.0) ? 0 : -sp->vz[s] * _pic_param.invD.z * f;
            rho += (1 - sp->rx[s]) * (1 - sp->ry[s]) * (1 - sp->rz[s]) * f;

        }
    };

    atomicAddReal(&fJx[s0/*                   */], J[0]);
    atomicAddReal(&fJx[s0 + _pic_param.strides.x], J[0]);
    atomicAddReal(&fJy[s0/*                   */], J[2]);
    atomicAddReal(&fJy[s0 + _pic_param.strides.y], J[3]);
    atomicAddReal(&fJz[s0/*                   */], J[4]);
    atomicAddReal(&fJz[s0 + _pic_param.strides.z], J[5]);
    atomicAddReal(&fRho[s0], rho);

};


int spParticleUpdateBorisYee(spParticle *sp, Real dt,
                             const struct spField_s *fE, const struct spField_s *fB,
                             struct spField_s *fRho, struct spField_s *fJ)
{
    if (sp == NULL) { return SP_DO_NOTHING; }


    Real *rho;
    Real *J[3];
    Real *E[3];
    Real *B[3];

    SP_CALL(spFieldSubArray((spField *) fRho, (void **) &rho));
    SP_CALL(spFieldSubArray((spField *) fJ, (void **) J));
    SP_CALL(spFieldSubArray((spField *) fE, (void **) E));
    SP_CALL(spFieldSubArray((spField *) fB, (void **) B));


    size_type grid_dim[3], block_dim[3];

    SP_CALL(spPICBorisSetupParam(sp, SP_DOMAIN_CENTER, grid_dim, block_dim));

    void **p_data;

    SP_CALL(spParticleGetAllAttributeData_device(sp, &p_data));

    SP_DEVICE_CALL_KERNEL(spParticleUpdateBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          dt, (boris_particle *) p_data, E[0], E[1], E[2], B[0], B[1], B[2]);

    SP_CALL(spParticleSync(sp));

    SP_DEVICE_CALL_KERNEL(spParticleAccumlateBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (boris_particle *) p_data, J[0], J[1], J[2], rho);

    SP_CALL(spFieldSync(fJ));


//    SP_CALL(spFieldSync(fRho));

    return SP_SUCCESS;
}


#endif //SIMPLA_SPBORIS_DEVICE_H
