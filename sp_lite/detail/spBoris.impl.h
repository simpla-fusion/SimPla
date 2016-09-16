//
// Created by salmon on 16-9-4.
//

#ifndef SIMPLA_SPBORIS_H
#define SIMPLA_SPBORIS_H

#include <math.h>
#include <assert.h>
#include "../sp_lite_def.h"
#include "../spPICBoris.h"
#include "../spParticle.h"
#include "../spMesh.h"
#include "../spField.h"
#include "../spRandom.h"
#include "../spPhysicalConstants.h"

#include "sp_device.h"
#include "../spParticle.impl.h"

typedef struct
{
    uint3 min;
    uint3 max;

    uint3 center_min;
    uint3 center_max;
    uint3 strides;
    uint3 g_strides;

    Real3 invD;

    Real charge;
    Real mass;
    Real cmr;

} _spPICBorisParam;

#define NULL_ID -1

__constant__ _spPICBorisParam
        _pic_param;

INLINE int spPICBorisSetupParam(spParticle *sp, int tag, size_type *grid_dim, size_type *block_dim)
{
    int error_code = SP_SUCCESS;
    _spPICBorisParam param;
    size_type min[3], max[3], strides[3];
    Real inv_dx[3];
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);
    SP_CALL(spMeshGetDomain(m, tag, min, max, NULL));
    SP_CALL(spMeshGetStrides(m, strides));
    SP_CALL(spMeshGetInvDx(m, inv_dx));

    param.min = sizeType2Dim3(min);
    param.max = sizeType2Dim3(max);
    param.strides = sizeType2Dim3(strides);
    param.invD = real2Real3(inv_dx);
    param.cmr = spParticleGetCharge(sp) / spParticleGetMass(sp);
    param.charge = spParticleGetCharge(sp);
    param.mass = spParticleGetMass(sp);


    spParallelMemcpyToSymbol(_pic_param, &param, sizeof(_spPICBorisParam));

    SP_CALL(spParallelThreadBlockDecompose(1, 3, min, max, grid_dim, block_dim));

    return error_code;
}

typedef struct boris_particle_p_s
{
    uint id;
    Real rx;
    Real ry;
    Real rz;
    Real vx;
    Real vy;
    Real vz;
    Real f;
    Real w;

} boris_p;


#define ll -0.5f
#define rr 0.5f

INLINE __device__

Real
cache_gather(Real const *f, Real rx, Real ry, Real rz)
{
    static const int s_c = 13, IX = 1, IY = 3, IZ = 9;

    return (f[s_c + IX + IY + IZ /*  */] * (rx - ll) * (ry - ll) * (rz - ll)
            + f[s_c + IX + IY /*     */] * (rx - ll) * (ry - ll) * (rr - rz)
            + f[s_c + IX + IZ /*     */] * (rx - ll) * (rr - ry) * (rz - ll)
            + f[s_c + IX /*          */] * (rx - ll) * (rr - ry) * (rr - rz)
            + f[s_c + IY + IZ /*     */] * (rr - rx) * (ry - ll) * (rz - ll)
            + f[s_c + IY /*          */] * (rr - rx) * (ry - ll) * (rr - rz)
            + f[s_c + IZ /*          */] * (rr - rx) * (rr - ry) * (rz - ll)
            + f[s_c + 0 /*           */] * (rr - rx) * (rr - ry) * (rr - rz));
}

INLINE __device__

void
cache_scatter(Real v, Real *f, Real rx, Real ry, Real rz)
{
    static const int s_c = 13, IX = 1, IY = 3, IZ = 9;
    f[s_c + IX + IY + IZ /**/] += (v * (rx - ll) * (ry - ll) * (rz - ll));
    f[s_c + IX + IY /*     */] += (v * (rx - ll) * (ry - ll) * (rr - rz));
    f[s_c + IX + IZ /*     */] += (v * (rx - ll) * (rr - ry) * (rz - ll));
    f[s_c + IX /*          */] += (v * (rx - ll) * (rr - ry) * (rr - rz));
    f[s_c + IY + IZ /*     */] += (v * (rr - rx) * (ry - ll) * (rz - ll));
    f[s_c + IY /*          */] += (v * (rr - rx) * (ry - ll) * (rr - rz));
    f[s_c + IZ /*          */] += (v * (rr - rx) * (rr - ry) * (rz - ll));
    f[s_c + 0 /*           */] += (v * (rr - rx) * (rr - ry) * (rr - rz));
}


#undef ll
#undef rr

__device__ INLINE

void
spParticlePopBoris(boris_particle *sp, size_type s, boris_p *p)
{
    p->id = sp->id[s];
    p->rx = sp->rx[s];
    p->ry = sp->ry[s];
    p->rz = sp->rz[s];
    p->vx = sp->vx[s];
    p->vy = sp->vy[s];
    p->vz = sp->vz[s];
    p->f = sp->f[s];
    p->w = sp->w[s];

}

__device__ INLINE

void
spParticlePushBoris(boris_particle *sp, size_type s, boris_p *p)
{
    sp->id[s] = p->id;

    sp->rx[s] = p->rx;
    sp->ry[s] = p->ry;
    sp->rz[s] = p->rz;

    sp->vx[s] = p->vx;
    sp->vy[s] = p->vy;
    sp->vz[s] = p->vz;

    sp->f[s] = p->f;
    sp->w[s] = p->w;
}

__device__ INLINE

void
spParticleMoveBoris(Real dt, boris_p *p, Real const *E, Real const *B)
{

    p->rx += p->vx * _pic_param.invD.x * dt;
    p->ry += p->vy * _pic_param.invD.y * dt;
    p->rz += p->vz * _pic_param.invD.z * dt;


    __register__
    Real ax, ay, az;
//    __register__ Real tx, ty, tz;
//    __register__ Real tt;

    ax = E[0];//cache_gather(E + 0, p->rx - 0.5f, p->ry, p->rz);
    ay = E[1];//cache_gather(E + 27, p->rx, p->ry - 0.5f, p->rz);
    az = E[2];//cache_gather(E + 54, p->rx, p->ry, p->rz - 0.5f);
//    tx = cache_gather(B + 0, p->rx, p->ry - 0.5f, p->rz - 0.5f);
//    ty = cache_gather(B + 27, p->rx - 0.5f, p->ry, p->rz - 0.5f);
//    tz = cache_gather(B + 54, p->rx - 0.5f, p->ry - 0.5f, p->rz);

    ax *= _pic_param.cmr * dt;
    ay *= _pic_param.cmr * dt;
    az *= _pic_param.cmr * dt;

//    tx *= _pic_param.cmr * dt;
//    ty *= _pic_param.cmr * dt;
//    tz *= _pic_param.cmr * dt;


    p->vx += ax;
    p->vy += ay;
    p->vz += az;

//    __register__ Real v_x, v_y, v_z;
//
//    v_x = p->vx + (p->vy * tz - p->vz * ty);
//    v_y = p->vy + (p->vz * tx - p->vx * tz);
//    v_z = p->vz + (p->vx * ty - p->vy * tx);
//
//    tt = 2 / (tx * tx + ty * ty + tz * tz + 1);
//
//    p->vx += ax + (v_y * tz - v_z * ty) * tt;
//    p->vy += ax + (v_z * tx - v_x * tz) * tt;
//    p->vz += ax + (v_x * ty - v_y * tx) * tt;

}





//__constant__ _spPICBorisParam _pic_param;


INLINE __device__

uint _spMeshHash(uint x, uint y, uint z)
{
    return __umul24(x, _pic_param.strides.x) +
           __umul24(y, _pic_param.strides.y) +
           __umul24(z, _pic_param.strides.z);
}

INLINE __device__

uint _spMeshGlobalHash(uint x, uint y, uint z)
{
    return __umul24(x, _pic_param.g_strides.x) +
           __umul24(y, _pic_param.g_strides.y) +
           __umul24(z, _pic_param.g_strides.z);
}

INLINE __device__

int _SPMeshInBox(uint x, uint y, uint z)
{
    return (_pic_param.min.x + x < _pic_param.max.x && _pic_param.min.y + y < _pic_param.max.y
            && _pic_param.min.z + z < _pic_param.max.z);
}

INLINE __device__

void
spParticleInitializeBoris(boris_particle *sp, size_type s, Real vT, Real f0)
{

    sp->f[s] = f0;
    sp->w[s] = 0.0;

    sp->vx[s] *= vT;
    sp->vy[s] *= vT;
    sp->vz[s] *= vT;
}

SP_DEVICE_DECLARE_KERNEL(spParticleInitializeBorisYeeKernel, boris_particle *sp,
                         Real vT, Real f0, int PIC)
{
    size_type threadId = threadIdx.x * blockDim.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.z;
    size_type x = _pic_param.min.x + blockIdx.x;
    size_type y = _pic_param.min.y + blockIdx.y;
    size_type z = _pic_param.min.z + blockIdx.z;


//    if (threadId < PIC)
//    {
//        size_type s0 = x * _pic_param.strides.x + y * _pic_param.strides.y + z * _pic_param.strides.z;
//
////#pragma unroll
//        for (size_type s = s0 * _pic_param.max_pic + threadIdx.x, se = s + PIC; s < se;
//             s += blockDim.x)
//        {
//            sp->id[s] = 0x3F;
//            sp->f[s] = f0;
//            sp->w[s] = 0.0;
//
//
//            sp->rx[s] -= 0.5;
//            sp->ry[s] -= 0.5;
//            sp->rz[s] -= 0.5;
//
//            sp->vx[s] *= vT;
//            sp->vy[s] *= vT;
//            sp->vz[s] *= vT;
////            spParticleInitializeBoris(sp, s, vT, f0);
//        }
//    }

}

int spParticleInitializeBorisYee(spParticle *sp, Real n0, Real T0)
{
    if (sp == NULL) { return SP_DO_NOTHING; }
    int error_code = SP_SUCCESS;

    SP_CALL(spParticleDeploy(sp));

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    int dist_type[6] = {SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_UNIFORM,
                        SP_RAND_NORMAL, SP_RAND_NORMAL, SP_RAND_NORMAL};

    SP_CALL(spParticleInitialize(sp, dist_type));
    SP_CALL(spParticleBuildBucket(sp));

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

    return error_code;
}

/******************************************************************************************/



SP_DEVICE_DECLARE_KERNEL (spParticleUpdateBorisYeeKernel,
                          boris_particle *sp,
                          size_type const *start_pos,
                          size_type const *end_pos,
                          size_type const *sorted_index,
                          Real dt,
                          Real const *Ex,
                          Real const *Ey,
                          Real const *Ez,
                          Real const *Bx,
                          Real const *By,
                          Real const *Bz)
{

    int s0 = _spMeshHash(_pic_param.min.x + blockIdx.x, _pic_param.min.y + blockIdx.y, _pic_param.min.z + blockIdx.z);

    __shared__ Real cE[6];
    __shared__ Real cB[6];


    if (threadIdx.x == 0)
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

//    __shared__  Real cE[27 * 3];
//    __shared__  Real cB[27 * 3];
//    __shared__ int dest_tail;
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
/**
 *   -1 -> 0b11
 *   00 -> 0b00
 *    1 -> 0b01
 *    (i+4)& 0x3
 */


    if (start_pos[s0] + threadIdx.x < end_pos[s0])
    {
        size_type s = sorted_index[start_pos[s0] + threadIdx.x];
        assert(sp->id[s] == s0);
        struct boris_particle_p_s p;

        spParticlePopBoris(sp, s, &p);

        Real E[3], B[3];

        E[0] = cE[0] * (0.5f - p.rx) + cE[1] * (0.5f + p.rx);
        E[1] = cE[2] * (0.5f - p.ry) + cE[3] * (0.5f + p.ry);
        E[2] = cE[4] * (0.5f - p.rz) + cE[5] * (0.5f + p.rz);

        spParticleMoveBoris(dt, &p, (Real const *) E, (Real const *) B);

        uint x = _pic_param.min.x + blockIdx.x + (int) (p.rx + 0.5);
        uint y = _pic_param.min.y + blockIdx.y + (int) (p.ry + 0.5);
        uint z = _pic_param.min.z + blockIdx.z + (int) (p.rz + 0.5);

        p.id = (x < _pic_param.center_min.x || x >= _pic_param.center_max.x
                || y < _pic_param.center_min.y || y >= _pic_param.center_max.y
                || z < _pic_param.center_min.z || z >= _pic_param.center_max.z) ? NULL_ID : _spMeshHash(x, y, z);

        p.rx -= (int) (p.rx + .5);
        p.ry -= (int) (p.ry + .5);
        p.rz -= (int) (p.rz + .5);

        spParticlePushBoris(sp, s, &p);

    }

};

SP_DEVICE_DECLARE_KERNEL (spParticleAccumlateBorisYeeKernel,
                          boris_particle *sp,
                          size_type const *start_pos,
                          size_type const *end_pos,
                          size_type const *particle_index,
                          Real *fJx,
                          Real *fJy,
                          Real *fJz,
                          Real * fRho
)
{

    uint x = _pic_param.min.x + blockIdx.x;
    uint y = _pic_param.min.y + blockIdx.y;
    uint z = _pic_param.min.z + blockIdx.z;

    int s0 = _spMeshHash(x, y, z);

    __shared__ Real J[6];
    __shared__ Real rho;

    if (threadIdx.x == 0)
    {
        rho = 0;
        for (int i = 0; i < 6; ++i) { J[i] = 0; }
    }

            spParallelSyncThreads();

    if (start_pos[s0] + threadIdx.x < end_pos[s0])
    {
        int s = particle_index[start_pos[s0] + threadIdx.x];

        assert(sp->id[s] == s0);

        Real f = sp->f[s];

        if (sp->rx[s] < 0.0) { atomicAdd(&J[0], sp->vx[s] * _pic_param.invD.x * f); }
        if (sp->rx[s] >= 0.0) { atomicAdd(&J[1], -sp->vx[s] * _pic_param.invD.x * f); }
        if (sp->ry[s] < 0.0) { atomicAdd(&J[2], sp->vy[s] * _pic_param.invD.y * f); }
        if (sp->ry[s] >= 0.0) { atomicAdd(&J[3], -sp->vy[s] * _pic_param.invD.y * f); }
        if (sp->rz[s] < 0.0) { atomicAdd(&J[4], sp->vz[s] * _pic_param.invD.z * f); }
        if (sp->rz[s] >= 0.0) { atomicAdd(&J[5], -sp->vz[s] * _pic_param.invD.z * f); }

        atomicAdd(&rho, (1 - sp->rx[s]) * (1 - sp->ry[s]) * (1 - sp->rz[s]) * f);

    };

    atomicAddReal(&fJx[s0/*                   */], J[0]);
    atomicAddReal(&fJx[s0 + _pic_param.strides.x], J[1]);
    atomicAddReal(&fJy[s0/*                   */], J[2]);
    atomicAddReal(&fJy[s0 + _pic_param.strides.y], J[3]);
    atomicAddReal(&fJz[s0/*                   */], J[4]);
    atomicAddReal(&fJz[s0 + _pic_param.strides.z], J[5]);
    atomicAddReal(&fRho[s0] /*                 */, rho
    );

};


int
spParticleUpdateBorisYee(spParticle *sp, Real dt,
                         const struct spField_s *fE, const struct spField_s *fB,
                         struct spField_s *fRho, struct spField_s *fJ)
{
    int error_code = SP_SUCCESS;

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

    size_type *start_pos, *end_pos, *index;

    SP_CALL(spParticleGetBucketIndex(sp, &start_pos, &end_pos, &index));

    SP_DEVICE_CALL_KERNEL(spParticleUpdateBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (boris_particle *) p_data,
                          start_pos, end_pos, index,
                          dt, E[0], E[1], E[2], B[0], B[1], B[2]);


    SP_CALL(spParticleSync(sp));


    SP_DEVICE_CALL_KERNEL(spParticleAccumlateBorisYeeKernel,
                          sizeType2Dim3(grid_dim),
                          sizeType2Dim3(block_dim),
                          (boris_particle *) p_data,
                          start_pos, end_pos, index,
                          J[0],
                          J[1],
                          J[2],
                          rho);

    SP_CALL(spFieldSync(fJ));
//    SP_CALL(spFieldSync(fRho));

    return error_code;
}


#endif //SIMPLA_SPBORIS_H
