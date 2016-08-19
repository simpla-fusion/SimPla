//
// Created by salmon on 16-8-4.
//

#ifndef SIMPLA_SPBORIS_DEVICE_H
#define SIMPLA_SPBORIS_DEVICE_H


#include "../sp_lite_def.h"

#include <math.h>

#include "../spMesh.h"
#include "../spField.h"
#include "../spPICBoris.h"
#include "../spRandom.h"
#include "../spPhysicalConstants.h"

#include "sp_device.h"
#include "spContext.impl.h"
#define ll 0
#define rr 1

INLINE __device__ Real
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

INLINE __device__ void
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

INLINE __device__   int _spMeshHash(int x, int y, int z)
{
    return __mul24(x, sp_ctx_d.strides.x) +
        __mul24(y, sp_ctx_d.strides.y) +
        __mul24(z, sp_ctx_d.strides.z);
}


__device__ void
spParticleInitializeBoris(boris_particle *sp, size_type s, Real vT, Real f0, int uniform_sample)
{
    sp->id[s] = 0;
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
                         boris_particle *sp, size_type num_pic,
                         Real vT, Real f0, int uniform_sample)
{


//    for (size_type x = min.x + blockIdx.x; x < max.x; x += gridDim.x)
//        for (size_type y = min.y + blockIdx.y; y < max.y; y += gridDim.y)
//            for (size_type z = min.z + blockIdx.z; z < max.z; z += gridDim.z)
//
    size_type threadId = threadIdx.x * blockDim.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.z;
    size_type num_of_thread = blockDim.x * blockDim.x * blockDim.x;

    size_type x = sp_ctx_d.min.x + blockIdx.x;
    size_type y = sp_ctx_d.min.y + blockIdx.y;
    size_type z = sp_ctx_d.min.z + blockIdx.z;


    if (threadId < num_pic)
    {
        size_type s0 = x * sp_ctx_d.strides.x + y * sp_ctx_d.strides.y + z * sp_ctx_d.strides.z;

#pragma unroll
        for (size_type s = s0 * num_pic + threadIdx.x, se = (s0 + 1) * num_pic; s < se; s += blockDim.x)
        {
            spParticleInitializeBoris(sp, s, vT, f0, uniform_sample);

        }
    }

}

int spParticleInitializeBorisYee(spParticle *sp, Real n0, Real T0, int do_important_sample)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    SP_CALL(spParticleDeploy(sp));

    size_type max_number_of_entities = spParticleGetNumberOfEntities(sp);

    int dist_type[6] =
        {SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_NORMAL, SP_RAND_NORMAL, SP_RAND_NORMAL};

    SP_CALL(spParticleInitialize(sp, dist_type));

    Real dx[3];

    SP_CALL(spMeshGetDx(m, dx));

    Real vT = (Real) sqrt(2.0 * SI_Boltzmann_constant * T0 / spParticleGetMass(sp));

    Real f0 = n0 * dx[0] * dx[1] * dx[2] / spParticleGetPIC(sp);

    size_type x_min[3], x_max[3], strides[3];

    SP_CALL(spMeshGetArrayShape(m, SP_DOMAIN_CENTER, x_min, x_max, strides));

    void **device_data;

    spParticleGetAllAttributeData_device(sp, &device_data, NULL);

    SP_DEVICE_CALL_KERNEL(spParticleInitializeBorisYeeKernel, spParallelDeviceGridDim(), spParallelDeviceBlockDim(),
                          (boris_particle *) device_data, spParticleGetMaxPIC(sp),
                          vT, f0, do_important_sample
    );

    spParallelDeviceFree((void **) &device_data);
    return SP_SUCCESS;
}

/******************************************************************************************/

typedef struct boris_particle_p_s
{
    Real rx;
    Real ry;
    Real rz;
    Real vx;
    Real vy;
    Real vz;
    Real f;
    Real w;

} boris_p;

__device__ INLINE void spParticlePopBoris(boris_particle *sp, size_type s, boris_p *p)
{
    p->rx = sp->rx[s];
    p->ry = sp->ry[s];
    p->rz = sp->rz[s];
    p->vx = sp->vx[s];
    p->vy = sp->vy[s];
    p->vz = sp->vz[s];
    p->f = sp->f[s];
    p->w = sp->w[s];
}


__device__ INLINE void spParticlePushBoris(boris_particle *sp, size_type s, boris_p const *p)
{

    sp->rx[s] = p->rx;
    sp->ry[s] = p->ry;
    sp->rz[s] = p->rz;
    sp->vx[s] = p->vx;
    sp->vy[s] = p->vy;
    sp->vz[s] = p->vz;
    sp->f[s] = p->f;
    sp->w[s] = p->w;
}

__device__ INLINE void spParticleMoveBoris(boris_p *p, Real dt, Real cmr, Real const *E, Real const *B)
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

    ax *= cmr * dt;
    ay *= cmr * dt;
    az *= cmr * dt;

    tx *= cmr * dt;
    ty *= cmr * dt;
    tz *= cmr * dt;

    p->rx += p->vx * 0.5 * sp_ctx_d.inv_dx.x * dt;
    p->ry += p->vy * 0.5 * sp_ctx_d.inv_dx.y * dt;
    p->rz += p->vz * 0.5 * sp_ctx_d.inv_dx.z * dt;

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

    p->rx += p->vx * 0.5 * sp_ctx_d.inv_dx.x * dt;
    p->ry += p->vy * 0.5 * sp_ctx_d.inv_dx.y * dt;
    p->rz += p->vz * 0.5 * sp_ctx_d.inv_dx.z * dt;

}

typedef struct spParticlePage_s
{
    struct spParticlePage_s *next;

    size_type cell_index;

    size_type head;

    size_type tail;

}

    spParticlePage;

SP_DEVICE_DECLARE_KERNEL (spParticleUpdateBorisYeeKernel,
                          boris_particle *sp_old,
                          boris_particle *sp_new,
                          size_type const *p_head,
                          size_type const *p_tail,
                          Real const *Ex,
                          Real const *Ey,
                          Real const *Ez,
                          Real const *Bx,
                          Real const *By,
                          Real const *Bz,
                          Real dt,
                          Real cmr)
{
    size_type threadId = threadIdx.x * blockDim.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.z;
    size_type num_of_thread = blockDim.x * blockDim.x * blockDim.x;

    size_type x = sp_ctx_d.min.x + blockIdx.x;
    size_type y = sp_ctx_d.min.y + blockIdx.y;
    size_type z = sp_ctx_d.min.z + blockIdx.z;


    __shared__ Real cE[27 * 3];
    __shared__ Real cB[27 * 3];
    __shared__ spParticlePage buckets[27];

    __shared__ size_type p_index[num_of_thread];

    size_type s0 = _spMeshHash(x, y, z);

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
                    size_type s2 = s0 + _spMeshHash(i, j, k);

                    cE[s1] = Ex[s2];
                    cB[s1] = Bx[s2];
                    cE[s1 + 27] = Ey[s2];
                    cB[s1 + 27] = By[s2];
                    cE[s1 + 27 * 2] = Ez[s2];
                    cB[s1 + 27 * 2] = Bz[s2];

                }
    }
    else if (threadId < 27 * 3)
    {
        size_type s2 = s0 + _spMeshHash(((threadId % 3) - 1), ((threadId / 3) % 3 - 1), ((threadId / 9) - 1));

        cE[threadId] = Ex[s2];
        cB[threadId] = Bx[s2];
        cE[threadId + 27] = Ey[s2];
        cB[threadId + 27] = By[s2];
        cE[threadId + 54] = Ez[s2];
        cB[threadId + 54] = Bz[s2];
    }

#ifdef __CUDACC__
    __syncthreads();
#endif

    if (p_head[s0] + threadId < p_tail[s0])
    {
        boris_p p;

        spParticlePopBoris(sp_old, p_index[p_head[s0] + threadId], &p);

        spParticleMoveBoris(&p, dt, cmr, (Real const *) cE, (Real const *) cB);

        p_index[p_head[s0] + threadId] = &p_index[p_head[s0] + threadId] - p_index;

        spParticlePushBoris(sp_new, p_index[p_head[s0] + threadId], &p);
    }

};

SP_DEVICE_DECLARE_KERNEL (spParticleGatherBorisYeeKernel,
                          boris_particle *sp, size_type num_pic,
                          Real *fJx, Real *fJy, Real *fJz, Real *fRho)
{
    size_type threadId = threadIdx.x * blockDim.x + threadIdx.y * blockDim.y + threadIdx.z * blockDim.z;
    size_type num_of_thread = blockDim.x * blockDim.x * blockDim.x;

    size_type x = sp_ctx_d.min.x + blockIdx.x;
    size_type y = sp_ctx_d.min.y + blockIdx.y;
    size_type z = sp_ctx_d.min.z + blockIdx.z;

    size_type s0 = _spMeshHash(x, y, z);

    Real Jx = 0, Jy = 0, Jz = 0, rho = 0;
    size_type s = s0 * num_pic + threadId;
    if (threadId < num_pic && sp->id[s] != 0)
    {
        Real w = sp->w[s] * sp->f[s]
            * (1 - sp->rx[s])
            * (1 - sp->ry[s])
            * (1 - sp->rz[s]);
        rho += w;
        Jx += w * sp->vx[s];
        Jy += w * sp->vy[s];
        Jz += w * sp->vz[s];

    }
    fJx[s0] += Jx;
    fJy[s0] += Jy;
    fJz[s0] += Jz;
    fRho[s0] += rho;

};


int spParticleUpdateBorisYee(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho, spField *fJ)
{

    void **p_current_data, **p_next_data;

    spParticleGetAllAttributeData_device(sp, &p_current_data, &p_next_data);

    Real *rho;
    Real *J[3];
    Real *E[3];
    Real *B[3];

    SP_CALL(spFieldSubArray((spField *) fRho, (void **) &rho));
    SP_CALL(spFieldSubArray((spField *) fJ, (void **) J));
    SP_CALL(spFieldSubArray((spField *) fE, (void **) E));
    SP_CALL(spFieldSubArray((spField *) fB, (void **) B));

    size_type max_pic = spParticleGetMaxPIC(sp);

    uint3 block_dim = {max_pic, 1, 1};

    SP_DEVICE_CALL_KERNEL(spParticleUpdateBorisYeeKernel, sp_ctx.grid_dim, block_dim,
                          (boris_particle *) p_current_data, (boris_particle *) p_next_data, max_pic,
                          E[0], E[1], E[2], B[0], B[1], B[2],
                          dt, spParticleGetCharge(sp) / spParticleGetMass(sp)
    );

    SP_DEVICE_CALL_KERNEL(spParticleGatherBorisYeeKernel, sp_ctx.grid_dim, block_dim,
                          (boris_particle *) p_current_data, max_pic, J[0], J[1], J[2], rho);

    SP_CALL(spParticleUpdate(sp));

    SP_CALL(spFieldSync(fJ));

    SP_CALL(spFieldSync(fRho));

    return SP_SUCCESS;
}


#endif //SIMPLA_SPBORIS_DEVICE_H
