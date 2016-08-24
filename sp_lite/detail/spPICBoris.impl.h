//
// Created by salmon on 16-8-4.
//

#ifndef SIMPLA_SPBORIS_DEVICE_H
#define SIMPLA_SPBORIS_DEVICE_H


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
    uint3 strides;
    Real3 inv_dx;

    int max_pic;

    Real charge;
    Real mass;
    Real cmr;

} _spPICBorisParam;

__constant__ _spPICBorisParam _pic_param;

int spPICBorisSetupParam(spParticle *sp, int tag, size_type *grid_dim, size_type *block_dim)
{
    _spPICBorisParam param;
    size_type min[3], max[3], strides[3];
    Real inv_dx[3];
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);
    SP_CALL(spMeshGetArrayShape(m, tag, min, max, strides));
    SP_CALL(spMeshGetInvDx(m, inv_dx));

    param.max_pic = spParticleGetMaxPIC(sp);

    param.min = sizeType2Dim3(min);
    param.max = sizeType2Dim3(max);
    param.strides = sizeType2Dim3(strides);
    param.inv_dx = real2Real3(inv_dx);
    param.cmr = spParticleGetCharge(sp) / spParticleGetMass(sp);
    param.charge = spParticleGetCharge(sp);
    param.mass = spParticleGetMass(sp);


    SP_CALL(spParallelMemcpyToCache(&_pic_param, &param, sizeof(_spPICBorisParam)));
    SP_CALL(spParallelThreadBlockDecompose(1, 3, min, max, grid_dim, block_dim));

    return SP_SUCCESS;
}


#define ll 0
#define rr 1

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

INLINE __device__

int _spMeshHash(int x, int y, int z)
{
    return __mul24(x, _pic_param.strides.x) +
           __mul24(y, _pic_param.strides.y) +
           __mul24(z, _pic_param.strides.z);
}

__device__ void
spParticleInitializeBoris(boris_particle *sp, size_type s, Real vT, Real f0, int uniform_sample)
{

    sp->f[s] = f0;
    sp->w[s] = 1.0;

    if (uniform_sample > 0)
    {
        sp->f[s] *= exp(-sp->vx[s] * sp->vx[s] - sp->vy[s] * sp->vy[s]
                        - sp->vz[s] * sp->vz[s]);
    }

    sp->vx[s] *= vT;
    sp->vy[s] *= vT;
    sp->vz[s] *= vT;
}

SP_DEVICE_DECLARE_KERNEL(spParticleInitializeBorisYeeKernel,
                         boris_particle *sp, Real vT, Real f0, int uniform_sample)
{


//    for (size_type x = min.x + blockIdx.x; x < max.x; x += gridDim.x)
//        for (size_type y = min.y + blockIdx.y; y < max.y; y += gridDim.y)
//            for (size_type z = min.z + blockIdx.z; z < max.z; z += gridDim.z)
//
    size_type threadId = threadIdx.x * blockDim.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.z;
//    size_type num_of_thread = blockDim.x * blockDim.x * blockDim.x;

    size_type x = _pic_param.min.x + blockIdx.x;
    size_type y = _pic_param.min.y + blockIdx.y;
    size_type z = _pic_param.min.z + blockIdx.z;


    if (threadId < _pic_param.max_pic)
    {
        size_type s0 = x * _pic_param.strides.x + y * _pic_param.strides.y + z * _pic_param.strides.z;

#pragma unroll
        for (size_type s = s0 * _pic_param.max_pic + threadIdx.x, se = (s0 + 1) * _pic_param.max_pic; s < se;
             s += blockDim.x)
        {
            sp->id[s] = 0;
            spParticleInitializeBoris(sp, s, vT, f0, uniform_sample);

        }
    }

}

int spParticleInitializeBorisYee(spParticle *sp, Real n0, Real T0, int do_important_sample)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    SP_CALL(spParticleDeploy(sp));

    int dist_type[6] =
            {SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_NORMAL, SP_RAND_NORMAL, SP_RAND_NORMAL};

    SP_CALL(spParticleInitialize(sp, dist_type));

    Real dx[3];

    SP_CALL(spMeshGetDx(m, dx));

    Real vT = (Real) sqrt(2.0 * SI_Boltzmann_constant * T0 / spParticleGetMass(sp));

    Real f0 = n0 * dx[0] * dx[1] * dx[2] / spParticleGetPIC(sp);

    void **device_data;

    SP_CALL(spParticleGetAllAttributeData_device(sp, &device_data));

    size_type grid_dim[3], block_dim[3];

    SP_CALL(spPICBorisSetupParam(sp, SP_DOMAIN_CENTER, grid_dim, block_dim));

    SP_DEVICE_CALL_KERNEL(spParticleInitializeBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (boris_particle *) device_data, vT, f0, do_important_sample);

    SP_CALL(spParticleSync(sp));

    return SP_SUCCESS;
}

/******************************************************************************************/

typedef struct boris_particle_p_s
{
    int id;
    Real rx;
    Real ry;
    Real rz;
    Real vx;
    Real vy;
    Real vz;
    Real f;
    Real w;

} boris_p;

__device__ INLINE void
spParticlePopBoris(boris_particle *sp, int s, boris_p *p)
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

__device__ INLINE void
spParticlePushBoris(boris_particle *sp, int s, boris_p const *p)
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

__device__ INLINE void
spParticleMoveBoris(Real dt, boris_p *p, Real const *E, Real const *B)
{
    __register__ Real ax, ay, az;
    __register__ Real tx, ty, tz;
    __register__ Real tt;

    ax = cache_gather(E + 0, p->rx - 0.5f, p->ry, p->rz);
    ay = cache_gather(E + 27, p->rx, p->ry - 0.5f, p->rz);
    az = cache_gather(E + 54, p->rx, p->ry, p->rz - 0.5f);
    tx = cache_gather(B + 0, p->rx, p->ry - 0.5f, p->rz - 0.5f);
    ty = cache_gather(B + 27, p->rx - 0.5f, p->ry, p->rz - 0.5f);
    tz = cache_gather(B + 54, p->rx - 0.5f, p->ry - 0.5f, p->rz);

    ax *= _pic_param.cmr * dt;
    ay *= _pic_param.cmr * dt;
    az *= _pic_param.cmr * dt;

    tx *= _pic_param.cmr * dt;
    ty *= _pic_param.cmr * dt;
    tz *= _pic_param.cmr * dt;

    p->rx += p->vx * 0.5 * _pic_param.inv_dx.x * dt;
    p->ry += p->vy * 0.5 * _pic_param.inv_dx.y * dt;
    p->rz += p->vz * 0.5 * _pic_param.inv_dx.z * dt;

    p->vx += ax;
    p->vy += ay;
    p->vz += az;

    __register__ Real v_x, v_y, v_z;

    v_x = p->vx + (p->vy * tz - p->vz * ty);
    v_y = p->vy + (p->vz * tx - p->vx * tz);
    v_z = p->vz + (p->vx * ty - p->vy * tx);

    tt = 2 / (tx * tx + ty * ty + tz * tz + 1);

    p->vx += ax + (v_y * tz - v_z * ty) * tt;
    p->vy += ax + (v_z * tx - v_x * tz) * tt;
    p->vz += ax + (v_x * ty - v_y * tx) * tt;

    p->rx += p->vx * 0.5 * _pic_param.inv_dx.x * dt;
    p->ry += p->vy * 0.5 * _pic_param.inv_dx.y * dt;
    p->rz += p->vz * 0.5 * _pic_param.inv_dx.z * dt;

}

SP_DEVICE_DECLARE_KERNEL (spParticleUpdateBorisYeeKernel, Real dt,
                          boris_particle *sp,
                          Real const *Ex,
                          Real const *Ey,
                          Real const *Ez,
                          Real const *Bx,
                          Real const *By,
                          Real const *Bz)
{
    int threadId = threadIdx.x * blockDim.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.z;
    int num_of_thread = blockDim.x * blockDim.x * blockDim.x;

    int x = _pic_param.min.x + blockIdx.x;
    int y = _pic_param.min.y + blockIdx.y;
    int z = _pic_param.min.z + blockIdx.z;


    __shared__  Real cE[27 * 3];
    __shared__  Real cB[27 * 3];


#ifdef __CUDACC__
    __syncthreads();
#endif


    if (num_of_thread < 27 * 3 && threadId == 0)
    {


        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k < 1; ++k)
                {
                    int s1 = 13 + i + j * 3 + k * 9;
                    int s2 = _spMeshHash(x + i, y + j, z + k);

                    cE[s1] = Ex[s2];
                    cB[s1] = Bx[s2];
                    cE[s1 + 27] = Ey[s2];
                    cB[s1 + 27] = By[s2];
                    cE[s1 + 27 * 2] = Ez[s2];
                    cB[s1 + 27 * 2] = Bz[s2];

                }
    } else if (threadId < 27 * 3)
    {

        int s2 = _spMeshHash(x + ((threadId % 3) - 1), y + ((threadId / 3) % 3 - 1), z + ((threadId / 9) - 1));

        cE[threadId] = Ex[s2];
        cB[threadId] = Bx[s2];
        cE[threadId + 27] = Ey[s2];
        cB[threadId + 27] = By[s2];
        cE[threadId + 54] = Ez[s2];
        cB[threadId + 54] = Bz[s2];
    }
    __shared__ int dest_tail;


#ifdef __CUDACC__
    __syncthreads();
#endif

    struct boris_particle_p_s p;
    int s0 = _spMeshHash(x, y, z) * _pic_param.max_pic;


    for (int src = threadId; src < _pic_param.max_pic; src += num_of_thread)
    {
        if ((sp->id[src] & 0x3F) == 0x0)
        {
            spParticlePopBoris(sp, s0 + src, &p);

            spParticleMoveBoris(dt, &p, (Real const *) cE, (Real const *) cB);

            spParticlePopBoris(sp, s0 + src, &p);

        }
    }

    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j)
            for (int k = -1; k < 1; ++k)
            {

                if (i == 0 && j == 0 && k == 0) { continue; }

                int s1 = _spMeshHash(x + i, y + j, z + k) * _pic_param.max_pic;

                /**
                 *   -1 -> 0b10
                 *   00 -> 0b00
                 *    1 -> 0b01
                 *    (i+3)%3
                 */
                int tag = 0;
                int flag = (i == 0 && j == 0 && k == 0) ? 0x0 : 0x3F;
                for (int src = threadId; src < _pic_param.max_pic; src += num_of_thread)
                {
                    if (sp->id[src] & 0x3F == tag)
                    {
                        spParticlePopBoris(sp, s1 + src, &p);

                        spParticleMoveBoris(dt, &p, (Real const *) cE, (Real const *) cB);

                        int dest = src;
                        if (s0 != s1)
                        {
                            while ((sp->id[s0 + (dest = atomicAddInt(&dest_tail, 1))] & 0x3F) != flag) {};
                        }


                        spParticlePopBoris(sp, s0 + dest, &p);

                    }
                }
            }
};

SP_DEVICE_DECLARE_KERNEL (spParticleGatherBorisYeeKernel,
                          boris_particle *sp,
                          Real *fJx,
                          Real *fJy,
                          Real *fJz,
                          Real *fRho)
{
    int threadId = threadIdx.x * blockDim.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.z;
    int num_of_thread = blockDim.x * blockDim.x * blockDim.x;

    int x = _pic_param.min.x + blockIdx.x;
    int y = _pic_param.min.y + blockIdx.y;
    int z = _pic_param.min.z + blockIdx.z;

    int s0 = _spMeshHash(x, y, z);

    Real Jx = 0, Jy = 0, Jz = 0, rho = 0;

    for (int s = s0 * _pic_param.max_pic + threadId; s < _pic_param.max_pic; s += num_of_thread)
    {
        if (sp->id[s] != 0) { continue; }

        Real w = sp->w[s] * sp->f[s]
                 * (1 - sp->rx[s])
                 * (1 - sp->ry[s])
                 * (1 - sp->rz[s]);
        rho += w;
        Jx += w * sp->vx[s];
        Jy += w * sp->vy[s];
        Jz += w * sp->vz[s];

    }
    atomicAddReal(&fJx[s0], Jx);
    atomicAddReal(&fJy[s0], Jy);
    atomicAddReal(&fJz[s0], Jz);
    atomicAddReal(&fRho[s0], rho);

};


int spParticleUpdateBorisYee(spParticle *sp,
                             Real dt,
                             const struct spField_s *fE,
                             const struct spField_s *fB,
                             struct spField_s *fRho,
                             struct spField_s *fJ)
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

    spPICBorisSetupParam(sp, SP_DOMAIN_CENTER, grid_dim, block_dim);

    void **p_data;

    spParticleGetAllAttributeData_device(sp, &p_data);

    SP_DEVICE_CALL_KERNEL(spParticleUpdateBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          dt, (boris_particle *) p_data, E[0], E[1], E[2], B[0], B[1], B[2]);

    SP_CALL(spParticleSync(sp));

    SP_DEVICE_CALL_KERNEL(spParticleGatherBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (boris_particle *) p_data, J[0], J[1], J[2], rho);

    SP_CALL(spFieldSync(fJ));

    SP_CALL(spFieldSync(fRho));

    return SP_SUCCESS;
}


#endif //SIMPLA_SPBORIS_DEVICE_H
