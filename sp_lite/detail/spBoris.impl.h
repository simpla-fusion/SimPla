//
// Created by salmon on 16-9-4.
//

#ifndef SIMPLA_SPBORIS_H
#define SIMPLA_SPBORIS_H

#include <math.h>
#include <assert.h>
#include "../sp_lite_def.h"
#include "../sp_config.h"
#include "../spPICBoris.h"
#include "../spParticle.h"
#include "../spMesh.h"
#include "../spField.h"
#include "../spPhysicalConstants.h"
#include "spParticle.impl.h"
#include "sp_device.h"
#include "spMesh.cu.h"
//
//typedef struct
//{
//    uint3 min;
//    uint3 max;
//    uint3 dims;
//    uint3 center_min;
//    uint3 center_max;
//    uint3 strides;
//    uint3 g_strides;
//
//    size_type num_of_cell;
//    size_type max_num_of_particle;
//    Real3 invD;
//
//    Real charge;
//    Real mass;
//    Real cmr;
//
//} _spPICBorisParam;
//
//#define NULL_ID (-1)
//
//__constant__ _spPICBorisParam _pic_param;
//
//INLINE int spPICBorisSetupParam(spParticle *sp)
//{
//    int error_code = SP_SUCCESS;
//    _spPICBorisParam param;
//    size_type min[3], max[3], strides[3], dims[3];
//    Real inv_dx[3];
//    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);
//
//    int iform = spMeshAttributeGetForm((spMeshAttribute *) sp);
//
//
//    SP_CALL(spMeshGetGlobalDims(m, dims));
//
//    SP_CALL(spMeshGetStrides(m, strides));
//
//    SP_CALL(spMeshGetInvDx(m, inv_dx));
//
//    assert(spParticleGetPIC(sp) < 256);
//
////    CHECK_INT(min[0]);
////    CHECK_INT(min[1]);
////    CHECK_INT(min[2]);
////    CHECK_INT(max[0]);
////    CHECK_INT(max[1]);
////    CHECK_INT(max[2]);
////    CHECK_INT(grid_dim[0]);
////    CHECK_INT(grid_dim[1]);
////    CHECK_INT(grid_dim[2]);
////    CHECK_INT(strides[0]);
////    CHECK_INT(strides[1]);
////    CHECK_INT(strides[2]);
//
//
//    param.num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);
//
//    param.max_num_of_particle = spParticleCapacity(sp);
//
//    param.min = sizeType2Dim3(min);
//
//    param.max = sizeType2Dim3(max);
//
//    param.dims = sizeType2Dim3(dims);
//
//    param.strides = sizeType2Dim3(strides);
//
//    size_type center_min[3], center_max[3];
//    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, center_min, center_max, NULL));
//
//    param.center_min = sizeType2Dim3(center_min);
//
//    param.center_max = sizeType2Dim3(center_max);
//
//    param.invD = real2Real3(inv_dx);
//
//    param.charge = spParticleGetCharge(sp);
//
//    param.mass = spParticleGetMass(sp);
//
//    param.cmr = param.charge / param.mass;
//
//    spParallelMemcpyToSymbol(_pic_param, &param, sizeof(_spPICBorisParam));
//
//    return error_code;
//}

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


SP_DEVICE_DECLARE_KERNEL(spParticleInitializeBorisYeeKernel, boris_particle *sp,
                         size_type const *start_pos,
                         size_type const *count,
                         size_type const *sorted_index,
                         Real vT, Real f0)
{

    uint s0 = SPMeshHash(_sp_mesh.center_min.x + blockIdx.x,
                         _sp_mesh.center_min.y + blockIdx.y,
                         _sp_mesh.center_min.z + blockIdx.z);
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

    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, NULL, NULL, grid_dim));

    block_dim[0] = NUMBER_OF_THREADS_PER_BLOCK;
    block_dim[1] = 1;
    block_dim[2] = 1;

    void **p_data;

    size_type *start_pos, *count, *sorted_idx;

    SP_CALL(spParticleGetAllAttributeData_device(sp, &p_data, NULL));

    SP_CALL(spParticleGetBucket(sp, &start_pos, &count, &sorted_idx, NULL));

    SP_CALL_DEVICE_KERNEL(spParticleInitializeBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (boris_particle *) p_data, start_pos, count, sorted_idx, vT, f0);

    SP_CALL(spParticleSync(sp));

    return SP_SUCCESS;
}

INLINE __device__ Real
cache_gather(Real const *f, Real rx, Real ry, Real rz)
{

    static const uint IX = 0x1 << 4, IY = 0x1 << 2, IZ = 0x1;
    int ix = (int) floor(rx);
    int iy = (int) floor(ry);
    int iz = (int) floor(rz);
    rx -= ix;
    ry -= iy;
    rz -= iz;

    int s_c = ((1 + ix) << 4) | ((1 + iy) << 2) | (1 + iz);

    static const Real ll = 0.0, rr = 1.0;
    return f[s_c + 0 /*           */] * (rx - ll) * (ry - ll) * (rz - ll) +
           f[s_c + IZ /*          */] * (rx - ll) * (ry - ll) * (rr - rz) +
           f[s_c + IY /*          */] * (rx - ll) * (rr - ry) * (rz - ll) +
           f[s_c + IY + IZ /*     */] * (rx - ll) * (rr - ry) * (rr - rz) +
           f[s_c + IX /*          */] * (rr - rx) * (ry - ll) * (rz - ll) +
           f[s_c + IX + IZ /*     */] * (rr - rx) * (ry - ll) * (rr - rz) +
           f[s_c + IX + IY /*     */] * (rr - rx) * (rr - ry) * (rz - ll) +
           f[s_c + IX + IY + IZ /**/] * (rr - rx) * (rr - ry) * (rr - rz);
}


INLINE __device__ void
cache_scatter(Real v, Real *f, Real rx, Real ry, Real rz)
{
    static const uint IX = 0x1 << 4, IY = 0x1 << 2, IZ = 0x1;
    int ix = (int) floor(rx);
    int iy = (int) floor(ry);
    int iz = (int) floor(rz);
    rx -= ix;
    ry -= iy;
    rz -= iz;

    int s_c = ((1 + ix) << 4) | ((1 + iy) << 2) | (1 + iz);
    static const Real ll = 0.0, rr = 1.0;
    atomicAdd(&f[s_c + 0 /*           */], (v * (rx - ll) * (ry - ll) * (rz - ll)));
    atomicAdd(&f[s_c + IZ /*          */], (v * (rx - ll) * (ry - ll) * (rr - rz)));
    atomicAdd(&f[s_c + IY /*          */], (v * (rx - ll) * (rr - ry) * (rz - ll)));
    atomicAdd(&f[s_c + IY + IZ /*     */], (v * (rx - ll) * (rr - ry) * (rr - rz)));
    atomicAdd(&f[s_c + IX /*          */], (v * (rr - rx) * (ry - ll) * (rz - ll)));
    atomicAdd(&f[s_c + IX + IZ /*     */], (v * (rr - rx) * (ry - ll) * (rr - rz)));
    atomicAdd(&f[s_c + IX + IY /*     */], (v * (rr - rx) * (rr - ry) * (rz - ll)));
    atomicAdd(&f[s_c + IX + IY + IZ /**/], (v * (rr - rx) * (rr - ry) * (rr - rz)));
}



/******************************************************************************************/


__device__ INLINE void
spParticleMoveBoris(Real dt, Real cmr, boris_p *p, Real const *E, Real const *B)
{

    p->rx += p->vx * _sp_mesh.inv_dx.x * dt * 0.5;
    p->ry += p->vy * _sp_mesh.inv_dx.y * dt * 0.5;
    p->rz += p->vz * _sp_mesh.inv_dx.z * dt * 0.5;

    __register__ Real ax, ay, az;
    __register__ Real tx, ty, tz;
//    __register__ Real tt;


    ax = cmr * dt * cache_gather(&(E[0]),   /**/ p->rx - (Real) 0.5, p->ry, p->rz);
    ay = cmr * dt * cache_gather(&(E[64]),  /**/ p->rx, p->ry - (Real) 0.5, p->rz);
    az = cmr * dt * cache_gather(&(E[128]), /**/ p->rx, p->ry, p->rz - (Real) 0.5);


    p->vx += ax * (Real) 0.5;
    p->vy += ay * (Real) 0.5;
    p->vz += az * (Real) 0.5;

//    __register__ Real v_x, v_y, v_z;
//
//    v_x = p->vx + (p->vy * tz - p->vz * ty);
//    v_y = p->vy + (p->vz * tx - p->vx * tz);
//    v_z = p->vz + (p->vx * ty - p->vy * tx);
//
//    tt = 2 / (tx * tx + ty * ty + tz * tz + 1);

    p->vx += ax * (Real) 0.5;//+ (v_y * tz - v_z * ty) * tt;
    p->vy += ax * (Real) 0.5;//+ (v_z * tx - v_x * tz) * tt;
    p->vz += ax * (Real) 0.5;//+ (v_x * ty - v_y * tx) * tt;

    p->rx += p->vx * _sp_mesh.inv_dx.x * dt * 0.5;
    p->ry += p->vy * _sp_mesh.inv_dx.y * dt * 0.5;
    p->rz += p->vz * _sp_mesh.inv_dx.z * dt * 0.5;
}

SP_DEVICE_DECLARE_KERNEL (spParticleUpdateBorisYeeKernel,
                          boris_particle *sp, size_type *cell_id,
                          size_type const *start_pos, size_type const *count, size_type const *sorted_index,
                          Real dt, Real cmr,
                          Real const *fE, Real const *fB)
{


    __shared__ Real cE[64 * 3];
    __shared__ Real cB[64 * 3];

    if (threadIdx.x < 64 * 3)
    {
        size_type s =
                (threadIdx.x >> 6) * _sp_mesh.num_of_cell +
                SPMeshHash((blockIdx.x + gridDim.x - 1 + ((threadIdx.x >> 4) & 0x3)) % gridDim.x,
                           (blockIdx.y + gridDim.y - 1 + ((threadIdx.x >> 2) & 0x3)) % gridDim.y,
                           (blockIdx.z + gridDim.z - 1 + ((threadIdx.x /**/) & 0x3)) % gridDim.z
                );

        assert(s < _sp_mesh.num_of_cell * 3);
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
    size_type s0 = SPMeshHash(blockIdx.x, blockIdx.y, blockIdx.z);


    if (threadIdx.x < count[s0])
    {
        size_type s = sorted_index[start_pos[s0] + threadIdx.x];

        struct boris_particle_p_s p;

        spParticlePopBoris(sp, s, &p);

        spParticleMoveBoris(dt, cmr, &p, (Real const *) cE, (Real const *) cB);

        int x = (int) floor(p.rx);
        int y = (int) floor(p.ry);
        int z = (int) floor(p.rz);

        p.rx -= x;
        p.ry -= y;
        p.rz -= z;

        x += blockIdx.x;
        y += blockIdx.y;
        z += blockIdx.z;

        cell_id[s] = SPMeshInCenter(x, y, z) ? SPMeshHash(x, y, z) : ((size_type) (-1));

        spParticlePushBoris(sp, s, &p);

    }

};

SP_DEVICE_DECLARE_KERNEL (spParticleAccumlateBorisYeeKernel,
                          boris_particle *sp,
                          size_type const *start_pos,
                          size_type const *count,
                          size_type const *sorted_idx,
                          Real charge, Real * fJ
)
{


    int s0 = SPMeshHash(blockIdx.x, blockIdx.y, blockIdx.z);

    __shared__ Real J[64 * 3];


    if (threadIdx.x < 64 * 3) { J[threadIdx.x] = 0; }

    spParallelSyncThreads();

    if (threadIdx.x < count[s0])
    {
        size_type s = sorted_idx[start_pos[s0] + threadIdx.x];

        Real f = charge * sp->f[s];
        Real rx = sp->rx[s];
        Real ry = sp->ry[s];
        Real rz = sp->rz[s];

        cache_scatter(f * sp->vx[s], &(J[0]),  /**/ rx - (Real) 0.5, ry, rz);
        cache_scatter(f * sp->vy[s], &(J[64]), /**/ rx, ry - (Real) 0.5, rz);
        cache_scatter(f * sp->vz[s], &(J[128]),/**/ rx, ry, rz - (Real) 0.5);
    };
    spParallelSyncThreads();

    if ((threadIdx.x < 64 * 3) && SPMeshInCenter(blockIdx.x, blockIdx.y, blockIdx.z))
    {
        size_type s = (threadIdx.x >> 6) * _sp_mesh.num_of_cell +
                      SPMeshHash(
                              (-1 + blockIdx.x + ((threadIdx.x >> 4) & 0x3)),
                              (-1 + blockIdx.y + ((threadIdx.x >> 2) & 0x3)),
                              (-1 + blockIdx.z + ((threadIdx.x /**/) & 0x3))
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
    SP_CALL(spMeshGetDims(m, grid_dim));
    block_dim[0] = SP_PARTICLE_DEFAULT_NUM_OF_PIC;
    block_dim[1] = 1;
    block_dim[2] = 1;

    void **current_data;

    size_type *start_pos, *count, *sorted_idx, *cell_hash;


    SP_CALL(spParticleGetAllAttributeData_device(sp, &current_data, NULL));

    SP_CALL(spParticleGetBucket(sp, &start_pos, &count, &sorted_idx, &cell_hash));

    SP_CALL_DEVICE_KERNEL(spParticleUpdateBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (boris_particle *) current_data, cell_hash, start_pos, count, sorted_idx,
                          dt, spParticleGetCharge(sp) / spParticleGetMass(sp), E, B);


    SP_CALL(spParticleSort(sp));

//    CHECK_INT(spParticleGlobalSize(sp));

    SP_CALL(spParticleSync(sp));


    SP_CALL_DEVICE_KERNEL(spParticleAccumlateBorisYeeKernel, sizeType2Dim3(grid_dim), sizeType2Dim3(block_dim),
                          (boris_particle *) current_data, start_pos, count, sorted_idx,
                          spParticleGetMass(sp), J);


    SP_CALL(spFieldSync(fJ));

    return SP_SUCCESS;
}


#endif //SIMPLA_SPBORIS_H
