//
// Created by salmon on 16-9-4.
//

#ifndef SIMPLA_SPBORIS_H
#define SIMPLA_SPBORIS_H

#include "../sp_lite_def.h"
#include "../spPICBoris.h"
#include "sp_device.h"

typedef struct
{
    uint3 min;
    uint3 max;
    uint3 strides;
    Real3 invD;

    int max_pic;

    Real charge;
    Real mass;
    Real cmr;

} _spPICBorisParam;
__constant__ _spPICBorisParam _pic_param;

INLINE int spPICBorisSetupParam(spParticle *sp, int tag, size_type *grid_dim, size_type *block_dim)
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
    param.invD = real2Real3(inv_dx);
    param.cmr = spParticleGetCharge(sp) / spParticleGetMass(sp);
    param.charge = spParticleGetCharge(sp);
    param.mass = spParticleGetMass(sp);


    spParallelMemcpyToSymbol(_pic_param, &param, sizeof(_spPICBorisParam));

    SP_CALL(spParallelThreadBlockDecompose(1, 3, min, max, grid_dim, block_dim));

    return SP_SUCCESS;
}

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


#define ll -0.5f
#define rr 0.5f

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

__device__ INLINE

void
spParticlePushBoris(boris_particle *sp, int s, boris_p *p)
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

#endif //SIMPLA_SPBORIS_H
