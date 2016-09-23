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
#include "../spPhysicalConstants.h"
#include "spParticle.impl.h"
#include "sp_device.h"

typedef struct
{
    uint3 min;
    uint3 max;
    uint3 dims;
    uint3 center_min;
    uint3 center_max;
    uint3 strides;
    uint3 g_strides;

    size_type num_of_cell;
    size_type max_num_of_particle;
    Real3 invD;

    Real charge;
    Real mass;
    Real cmr;

} _spPICBorisParam;

#define NULL_ID (-1)

__constant__ _spPICBorisParam _pic_param;

INLINE int spPICBorisSetupParam(spParticle *sp, int tag, size_type *grid_dim, size_type *block_dim)
{
    int error_code = SP_SUCCESS;
    _spPICBorisParam param;
    size_type min[3], max[3], strides[3], dims[3];
    Real inv_dx[3];
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    int iform = spMeshAttributeGetForm((spMeshAttribute *) sp);

    SP_CALL(spMeshGetDomain(m, tag, min, max, grid_dim));

    SP_CALL(spMeshGetDims(m, dims));

    SP_CALL(spMeshGetStrides(m, strides));

    SP_CALL(spMeshGetInvDx(m, inv_dx));

    assert(spParticleGetPIC(sp) < 256);

//    CHECK_INT(min[0]);
//    CHECK_INT(min[1]);
//    CHECK_INT(min[2]);
//    CHECK_INT(max[0]);
//    CHECK_INT(max[1]);
//    CHECK_INT(max[2]);
//    CHECK_INT(grid_dim[0]);
//    CHECK_INT(grid_dim[1]);
//    CHECK_INT(grid_dim[2]);
//    CHECK_INT(strides[0]);
//    CHECK_INT(strides[1]);
//    CHECK_INT(strides[2]);

    block_dim[0] = 256;
    block_dim[1] = 1;
    block_dim[2] = 1;

    param.num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);

    param.max_num_of_particle = spParticleCapacity(sp);

    param.min = sizeType2Dim3(min);

    param.max = sizeType2Dim3(max);

    param.dims = sizeType2Dim3(dims);

    param.strides = sizeType2Dim3(strides);

    size_type center_min[3], center_max[3];
    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, center_min, center_max, NULL));

    param.center_min = sizeType2Dim3(center_min);

    param.center_max = sizeType2Dim3(center_max);

    param.invD = real2Real3(inv_dx);

    param.charge = spParticleGetCharge(sp);

    param.mass = spParticleGetMass(sp);

    param.cmr = param.charge / param.mass;

    spParallelMemcpyToSymbol(_pic_param, &param, sizeof(_spPICBorisParam));


    return error_code;
}

typedef struct boris_particle_p_s
{
    size_type id;
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
spParticlePopBoris(boris_particle *sp, size_type s, boris_p *p)
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

__device__ INLINE  void
spParticlePushBoris(boris_particle *sp, size_type s, boris_p *p)
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


INLINE __device__ uint
_spMeshHash(uint x, uint y, uint z)
{
    return __umul24(x, _pic_param.strides.x) + __umul24(y, _pic_param.strides.y) + __umul24(z, _pic_param.strides.z);
}


SP_DEVICE_DECLARE_KERNEL(spParticleInitializeBorisYeeKernel, boris_particle *sp,
                         size_type const *start_pos,
                         size_type const *count,
                         size_type const *sorted_index,
                         Real vT, Real f0)
{

    uint s0 = _spMeshHash(_pic_param.min.x + blockIdx.x,
                          _pic_param.min.y + blockIdx.y,
                          _pic_param.min.z + blockIdx.z);
    if (threadIdx.x < count[s0])
    {
        size_type s = sorted_index[start_pos[s0] + threadIdx.x];

        sp->f[s] = f0;
        sp->w[s] = 0.0;

        sp->vx[s] *= vT;
        sp->vy[s] *= vT;
        sp->vz[s] *= vT;

    }

}

int spParticleInitializeBorisYee(spParticle *sp, Real n0, Real T0)
{
    if (sp == NULL) { return SP_DO_NOTHING; }


    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    int dist_type[6] = {SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_UNIFORM,
                        SP_RAND_NORMAL, SP_RAND_NORMAL, SP_RAND_NORMAL};

    SP_CALL(spParticleInitialize(sp, dist_type));

    Real dx[3];

    SP_CALL(spMeshGetDx(m, dx));

    Real vT = (Real) sqrt(2.0 * SI_Boltzmann_constant * T0 / spParticleGetMass(sp));

    Real f0 = n0 / spParticleGetPIC(sp) * spParticleGetCharge(sp);

    size_type grid_dim[3], block_dim[3];

    void **p_data;

    size_type *start_pos, *count, *sorted_idx;

    SP_CALL(spPICBorisSetupParam(sp, SP_DOMAIN_CENTER, grid_dim, block_dim));

    SP_CALL(spParticleGetAllAttributeData_device(sp, &p_data, NULL));

    SP_CALL(spParticleGetBucket(sp, &start_pos, &count, &sorted_idx, NULL));

    SP_CALL_DEVICE_KERNEL(spParticleInitializeBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (boris_particle *) p_data, start_pos, count, sorted_idx, vT, f0);

    SP_CALL(spParticleSync(sp));

    return SP_SUCCESS;
}

/******************************************************************************************/


__device__ INLINE void
spParticleMoveBoris(Real dt, boris_p *p, Real const *E, Real const *B)
{

//    p->rx += p->vx * _pic_param.invD.x * dt * 0.5;
//    p->ry += p->vy * _pic_param.invD.y * dt * 0.5;
//    p->rz += p->vz * _pic_param.invD.z * dt * 0.5;

    __register__ Real ax, ay, az;
//    __register__ Real tx, ty, tz;
//    __register__ Real tt;

    ax = E[0];//cache_gather(E + 0, p->rx - 0.5f, p->ry, p->rz);
    ay = E[1];//cache_gather(E + 27, p->rx, p->ry - 0.5f, p->rz);
    az = E[2];//cache_gather(E + 54, p->rx, p->ry, p->rz - 0.5f);

//    tx = B[0];
//    ty = B[1];
//    tz = B[2];
//    tx = cache_gather(B + 0, p->rx, p->ry - 0.5f, p->rz - 0.5f);
//    ty = cache_gather(B + 27, p->rx - 0.5f, p->ry, p->rz - 0.5f);
//    tz = cache_gather(B + 54, p->rx - 0.5f, p->ry - 0.5f, p->rz);
//
    ax *= _pic_param.cmr * dt * 0.5;
    ay *= _pic_param.cmr * dt * 0.5;
    az *= _pic_param.cmr * dt * 0.5;

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

    p->vx += ax;//+ (v_y * tz - v_z * ty) * tt;
    p->vy += ax;//+ (v_z * tx - v_x * tz) * tt;
    p->vz += ax;//+ (v_x * ty - v_y * tx) * tt;

//    p->rx += p->vx * _pic_param.invD.x * dt * 0.5;
//    p->ry += p->vy * _pic_param.invD.y * dt * 0.5;
//    p->rz += p->vz * _pic_param.invD.z * dt * 0.5;
}

INLINE __device__ Real
cache_gather(uint s_c, Real const *f, Real rx, Real ry, Real rz)
{

    uint IX = 0x1 << 4, IY = 0x1 << 2, IZ = 0x1;

    Real ll = 0.0, rr = 1.0;
    return f[s_c + IX + IY + IZ /**/] * (rx - ll) * (ry - ll) * (rz - ll) +
           f[s_c + IX + IY /*     */] * (rx - ll) * (ry - ll) * (rr - rz) +
           f[s_c + IX + IZ /*     */] * (rx - ll) * (rr - ry) * (rz - ll) +
           f[s_c + IX /*          */] * (rx - ll) * (rr - ry) * (rr - rz) +
           f[s_c + IY + IZ /*     */] * (rr - rx) * (ry - ll) * (rz - ll) +
           f[s_c + IY /*          */] * (rr - rx) * (ry - ll) * (rr - rz) +
           f[s_c + IZ /*          */] * (rr - rx) * (rr - ry) * (rz - ll) +
           f[s_c + 0 /*           */] * (rr - rx) * (rr - ry) * (rr - rz);
}


INLINE __device__ void
cache_scatter(uint s_c, Real v, Real *f, Real rx, Real ry, Real rz)
{
    Real ll = 0.0, rr = 1.0;

    static const uint IX = 0x1 << 4, IY = 0x1 << 2, IZ = 1;

    atomicAdd(&f[s_c + IX + IY + IZ /**/], (v * (rx - ll) * (ry - ll) * (rz - ll)));
    atomicAdd(&f[s_c + IX + IY /*     */], (v * (rx - ll) * (ry - ll) * (rr - rz)));
    atomicAdd(&f[s_c + IX + IZ /*     */], (v * (rx - ll) * (rr - ry) * (rz - ll)));
    atomicAdd(&f[s_c + IX /*          */], (v * (rx - ll) * (rr - ry) * (rr - rz)));
    atomicAdd(&f[s_c + IY + IZ /*     */], (v * (rr - rx) * (ry - ll) * (rz - ll)));
    atomicAdd(&f[s_c + IY /*          */], (v * (rr - rx) * (ry - ll) * (rr - rz)));
    atomicAdd(&f[s_c + IZ /*          */], (v * (rr - rx) * (rr - ry) * (rz - ll)));
    atomicAdd(&f[s_c + 0 /*           */], (v * (rr - rx) * (rr - ry) * (rr - rz)));
}
//
//
//#undef ll
//#undef rr

SP_DEVICE_DECLARE_KERNEL (spParticleUpdateBorisYeeKernel,
                          boris_particle *sp, size_type *cell_id,
                          size_type const *start_pos, size_type const *count, size_type const *sorted_index, Real dt,
                          Real const *fE, Real const *fB)
{


    __shared__ Real cE[64 * 3];
    __shared__ Real cB[64 * 3];

    assert(blockDim.x >= 64 * 3);
    if (threadIdx.x < 64 * 3)
    {
        size_type s =
                (threadIdx.x >> 6) * _pic_param.num_of_cell +
                _spMeshHash(_pic_param.min.x + (blockIdx.x + gridDim.x - 1 + ((threadIdx.x >> 4) & 0x3)) % gridDim.x,
                            _pic_param.min.y + (blockIdx.y + gridDim.y - 1 + ((threadIdx.x >> 2) & 0x3)) % gridDim.y,
                            _pic_param.min.z + (blockIdx.z + gridDim.z - 1 + ((threadIdx.x /**/) & 0x3)) % gridDim.z
                );

        assert(s < _pic_param.num_of_cell * 3);

        cE[threadIdx.x] = fE[s];
        cB[threadIdx.x] = fB[s];

    }


    spParallelSyncThreads();
/**
 *   -1 -> 0b11
 *   00 -> 0b00
 *    1 -> 0b01
 *    (i+4)& 0x3
 */
    size_type s0 = _spMeshHash(_pic_param.min.x + blockIdx.x,
                               _pic_param.min.y + blockIdx.y,
                               _pic_param.min.z + blockIdx.z);


    if (threadIdx.x < count[s0])
    {
        size_type s = sorted_index[start_pos[s0] + threadIdx.x];

        struct boris_particle_p_s p;

        spParticlePopBoris(sp, s, &p);

        Real E[3], B[3];

//        E[0] = cache_gather((uint) (p.rx + 0.5) << 4, &(cE[s0]/*       */), p.rx, p.ry, p.rz);
//
//        E[1] = cache_gather((uint) (p.ry + 0.5) << 2, &(cE[s0 + 0x1 << 6]), p.rx, p.ry, p.rz);
//
//        E[2] = cache_gather((uint) (p.rz + 0.5)/* */, &(cE[s0 + 0x2 << 6]), p.rx, p.ry, p.rz);
//
//
//        B[0] = cache_gather(((uint) (p.ry + 0.5) << 2) | ((uint) (p.rz + 0.5)/* */),
//                            &(cB[s0]/*       */), p.rx, p.ry, p.rz);
//        B[1] = cache_gather(((uint) (p.rx + 0.5) << 4) | ((uint) (p.rz + 0.5)/* */),
//                            &(cB[s0 + 0x1 << 6]), p.rx, p.ry, p.rz);
//        B[2] = cache_gather(((uint) (p.rx + 0.5) << 4) | ((uint) (p.ry + 0.5) << 2),
//                            &(cB[s0 + 0x2 << 6]), p.rx, p.ry, p.rz);

//        spParticleMoveBoris(dt, &p, (Real const *) E, (Real const *) B);

        int x = (int) floor(p.rx);
        int y = (int) floor(p.ry);
        int z = (int) floor(p.rz);

        p.rx -= x;
        p.ry -= y;
        p.rz -= z;

        x = _pic_param.min.x + blockIdx.x + x;
        y = _pic_param.min.y + blockIdx.y + y;
        z = _pic_param.min.z + blockIdx.z + z;

        cell_id[s] = (x >= _pic_param.center_min.x && x < _pic_param.center_max.x &&
                      y >= _pic_param.center_min.y && y < _pic_param.center_max.y &&
                      z >= _pic_param.center_min.z && z < _pic_param.center_max.z) ?
                     _spMeshHash(x, y, z) : ((size_type) (-1));

        spParticlePushBoris(sp, s, &p);

    }

};

SP_DEVICE_DECLARE_KERNEL (spParticleAccumlateBorisYeeKernel,
                          boris_particle *sp,
                          size_type const *start_pos,
                          size_type const *count,
                          size_type const *sorted_idx,
                          Real * fJ
)
{

    uint x = _pic_param.min.x + blockIdx.x;
    uint y = _pic_param.min.y + blockIdx.y;
    uint z = _pic_param.min.z + blockIdx.z;

    int s0 = _spMeshHash(x, y, z);

    __shared__ Real J[64 * 3];


    if (threadIdx.x < 64 * 3) { J[threadIdx.x] = 0; }

    spParallelSyncThreads();

    if (threadIdx.x < count[s0])
    {
        size_type s = sorted_idx[start_pos[s0] + threadIdx.x];


        Real f = sp->f[s] * sp->vx[s];
        Real rx = sp->rx[s];
        Real ry = sp->ry[s];
        Real rz = sp->rz[s];

        cache_scatter((uint) (rx + 0.5) << 4, f, &(J[0]/*       */), rx, ry, rz);
        cache_scatter((uint) (ry + 0.5) << 2, f, &(J[0 + 0x1 << 6]), rx, ry, rz);
        cache_scatter((uint) (rz + 0.5)/* */, f, &(J[0 + 0x2 << 6]), rx, ry, rz);
    };
    spParallelSyncThreads();

    if ((threadIdx.x < 64 * 3) &&
        (x >= _pic_param.center_min.x && x <= _pic_param.center_max.x &&
         y >= _pic_param.center_min.y && y <= _pic_param.center_max.y &&
         z >= _pic_param.center_min.z && z <= _pic_param.center_max.z))
    {
        size_type s = (threadIdx.x >> 6) * _pic_param.num_of_cell +
                      _spMeshHash(
                              _pic_param.min.x + (blockIdx.x - 1 + ((threadIdx.x >> 4) & 0x3) + gridDim.x) % gridDim.x,
                              _pic_param.min.y + (blockIdx.y - 1 + ((threadIdx.x >> 2) & 0x3) + gridDim.y) % gridDim.y,
                              _pic_param.min.z + (blockIdx.z - 1 + ((threadIdx.x /**/) & 0x3) + gridDim.z) % gridDim.z
                      );

        atomicAdd(&fJ[s], J[threadIdx.x]);
    }

};


int
spParticleUpdateBorisYee(spParticle *sp, Real dt, const spField *fE, const spField *fB,
                         spField *fRho, spField *fJ)
{


    if (sp == NULL) { return SP_FAILED; }


    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    Real *J = (Real *) spFieldData(fJ);
    Real *E = (Real *) spFieldData((spField *) fE);
    Real *B = (Real *) spFieldData((spField *) fB);


    size_type grid_dim[3], block_dim[3];

    void **current_data;

    size_type *start_pos, *count, *sorted_idx, *cell_hash;

    SP_CALL(spPICBorisSetupParam(sp, SP_DOMAIN_ALL, grid_dim, block_dim));

    SP_CALL(spParticleGetAllAttributeData_device(sp, &current_data, NULL));

    SP_CALL(spParticleGetBucket(sp, &start_pos, &count, &sorted_idx, &cell_hash));

    SP_CALL_DEVICE_KERNEL(spParticleUpdateBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (boris_particle *) current_data, cell_hash, start_pos, count, sorted_idx, dt, E, B)


    SP_CALL(spParticleSort(sp));

//    CHECK_INT(spParticleGlobalSize(sp));

    SP_CALL(spParticleSync(sp));


    SP_CALL_DEVICE_KERNEL(spParticleAccumlateBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (boris_particle *) current_data, start_pos, count, sorted_idx, J)
//

    SP_CALL(spFieldSync(fJ));

    return SP_SUCCESS;
}


#endif //SIMPLA_SPBORIS_H
