//
// Created by salmon on 16-8-4.
//

#ifndef SIMPLA_SPBORIS_DEVICE_H
#define SIMPLA_SPBORIS_DEVICE_H


#include "../sp_lite_def.h"
#include "../spPICBoris.h"
#include "sp_device.h"
#include "../spRandom.h"
#include <math.h>

#define ll 0
#define rr 1


INLINE __device__ void
cache_gather(Real *v,
             Real const *f,
             size_type s_c,
             size_type IX,
             size_type IY,
             size_type IZ,
             Real rx,
             Real ry,
             Real rz)
{

    *v = *v
        + (f[s_c + IX + IY + IZ /*  */] * (rx - ll) * (ry - ll) * (rz - ll)
            + f[s_c + IX + IY /*     */] * (rx - ll) * (ry - ll) * (rr - rz)
            + f[s_c + IX + IZ /*     */] * (rx - ll) * (rr - ry) * (rz - ll)
            + f[s_c + IX /*          */] * (rx - ll) * (rr - ry) * (rr - rz)
            + f[s_c + IY + IZ /*     */] * (rr - rx) * (ry - ll) * (rz - ll)
            + f[s_c + IY /*          */] * (rr - rx) * (ry - ll) * (rr - rz)
            + f[s_c + IZ /*          */] * (rr - rx) * (rr - ry) * (rz - ll)
            + f[s_c + 0 /*           */] * (rr - rx) * (rr - ry) * (rr - rz));
}

INLINE __device__ void
cache_scatter(Real v, Real *f, Real rx, Real ry, Real rz, size_type s_c)
{
//    atomicAdd(&f[s_c + IX + IY + IZ /**/], (v * (rx - ll) * (ry - ll) * (rz - ll)));
//    atomicAdd(&f[s_c + IX + IY /*     */], (v * (rx - ll) * (ry - ll) * (rr - rz)));
//    atomicAdd(&f[s_c + IX + IZ /*     */], (v * (rx - ll) * (rr - ry) * (rz - ll)));
//    atomicAdd(&f[s_c + IX /*          */], (v * (rx - ll) * (rr - ry) * (rr - rz)));
//    atomicAdd(&f[s_c + IY + IZ /*     */], (v * (rr - rx) * (ry - ll) * (rz - ll)));
//    atomicAdd(&f[s_c + IY /*          */], (v * (rr - rx) * (ry - ll) * (rr - rz)));
//    atomicAdd(&f[s_c + IZ /*          */], (v * (rr - rx) * (rr - ry) * (rz - ll)));
//    atomicAdd(&f[s_c + 0 /*           */], (v * (rr - rx) * (rr - ry) * (rr - rz)));
}


#undef ll
#undef rr

#undef  s_c

INLINE __device__ void spBoris(Real cmr_dt, Real3 mesh_inv_dv,
                               size_type s, size_type IX, size_type IY, size_type IZ,
                               Real *rho, Real *Jx, Real *Jy, Real *Jz,
                               const Real *Ex, const Real *Ey, const Real *Ez,
                               const Real *Bx, const Real *By, const Real *Bz,
                               Real *rx, Real *ry, Real *rz,
                               Real *vx, Real *vy, Real *vz,
                               Real *f, Real *w)
{
    Real ax, ay, az;
    Real tx, ty, tz;

    Real tt;

    cache_gather(&ax, Ex, s, IX, IY, IZ, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
    cache_gather(&ay, Ey, s, IX, IY, IZ, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
    cache_gather(&az, Ez, s, IX, IY, IZ, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);
    cache_gather(&tx, Bx, s, IX, IY, IZ, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
    cache_gather(&ty, By, s, IX, IY, IZ, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
    cache_gather(&tz, Bz, s, IX, IY, IZ, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);

    ax *= cmr_dt;
    ay *= cmr_dt;
    az *= cmr_dt;

    tx *= cmr_dt;
    ty *= cmr_dt;
    tz *= cmr_dt;

    *rx += *vx * 0.5 * mesh_inv_dv.x;
    *ry += *vy * 0.5 * mesh_inv_dv.y;
    *rz += *vz * 0.5 * mesh_inv_dv.z;

    *vx += ax;
    *vy += ay;
    *vz += az;

    Real v_x, v_y, v_z;

    v_x = *vx + (*vy * tz - *vz * ty);
    v_y = *vy + (*vz * tx - *vx * tz);
    v_z = *vz + (*vx * ty - *vy * tx);

    tt = 2 / (tx * tx + ty * ty + tz * tz + 1);

    *vx += ax + (v_y * tz - v_z * ty) * tt;
    *vy += ax + (v_z * tx - v_x * tz) * tt;
    *vz += ax + (v_x * ty - v_y * tx) * tt;

    *rx += *vx * 0.5 * mesh_inv_dv.x;
    *ry += *vy * 0.5 * mesh_inv_dv.y;
    *rz += *vz * 0.5 * mesh_inv_dv.z;

    cache_scatter((*f) * (*w), rho, *rx, *ry, *rz, 0);
    cache_scatter((*f) * (*w)  /*   */* (*vx), Jx, *rx, *ry, *rz, 0);
    cache_scatter((*f) * (*w)  /*   */* (*vy), Jy, *rx, *ry, *rz, 0);
    cache_scatter((*f) * (*w)  /*   */* (*vz), Jz, *rx, *ry, *rz, 0);
}

SP_DEVICE_DECLARE_KERNEL(spParticleInitializeBorisYeeKernel,
                         boris_particle *sp, dim3 min, dim3 max, dim3 strides, size_type pic, Real vT,
                         Real f0, int uniform_sample)
{
    size_type threadId = (threadIdx.x) +
        (threadIdx.y) * blockDim.x +
        (threadIdx.z) * blockDim.x * blockDim.y;

    size_type num_of_thread = blockDim.z * blockDim.x * blockDim.y;
    for (int x = blockIdx.x + min.x; x < max.x; x += gridDim.x)
        for (int y = blockIdx.y + min.y; y < max.y; y += gridDim.y)
            for (int z = blockIdx.z + min.z; z < max.z; z += gridDim.z)
            {
                size_type s0 = threadId + x * strides.x + y * strides.y + z * strides.z;

                for (size_type s = 0; s < pic; s += num_of_thread)
                {
                    sp->id[s0 + s] = 0;
                    sp->f[s0 + s] = f0;
                    sp->w[s0 + s] = 1.0;
                    if (uniform_sample > 0)
                    {
                        sp->f[s0 + s] *= exp(-sp->vx[s0 + s] * sp->vx[s0 + s]
                                                 - sp->vy[s0 + s] * sp->vy[s0 + s]
                                                 - sp->vz[s0 + s] * sp->vz[s0 + s]
                        );
                    }

                    sp->vx[s0 + s] *= vT;
                    sp->vy[s0 + s] *= vT;
                    sp->vz[s0 + s] *= vT;

                }
            }

}

int spParticleInitializeBorisYee(spParticle *sp, Real n0, Real T0, int do_import_sample)
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
    strides[0] *= spParticleGetMaxPIC(sp);
    strides[1] *= spParticleGetMaxPIC(sp);
    strides[2] *= spParticleGetMaxPIC(sp);

    size_type blocks[3] = {16, 1, 1};
    size_type threads[3] = {SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE, 1, 1};

    void **device_data;

    spParticleGetAllAttributeData_device(sp, &device_data);

    SP_DEVICE_CALL_KERNEL(spParticleInitializeBorisYeeKernel,
                          sizeType2Dim3(blocks), sizeType2Dim3(threads),
                          (boris_particle *) device_data,
                          sizeType2Dim3(x_min), sizeType2Dim3(x_max), sizeType2Dim3(strides),
                          spParticleGetPIC(sp), vT, f0, do_import_sample
    );

    spParallelDeviceFree((void **) &device_data);
    return SP_SUCCESS;
}




/******************************************************************************************/
#define SP_MAX_NUM_OF_PARTICLE_ATTR 32

typedef struct
{
    Real inv_dv[3];
    Real cmr_dt;
    int max_pic;
    void *data[SP_MAX_NUM_OF_PARTICLE_ATTR];
    Real *rho;
    Real *J[3];
    Real *E[3];
    Real *B[3];

    size_type min[3];
    size_type max[3];
    size_type strides[3];

} boris_update_param;

struct boris_p_s
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
};

__constant__ boris_update_param g_boris_param;

#define  SP_CONSTANT_FIELD_SIZE   0x800

__constant__ Real Ex[SP_CONSTANT_FIELD_SIZE];

__constant__ Real Ey[SP_CONSTANT_FIELD_SIZE];

__constant__ Real Ez[SP_CONSTANT_FIELD_SIZE];

__constant__ Real Bx[SP_CONSTANT_FIELD_SIZE];

__constant__ Real By[SP_CONSTANT_FIELD_SIZE];

__constant__ Real Bz[SP_CONSTANT_FIELD_SIZE];

SP_DEVICE_DECLARE_KERNEL (spBorisYeeUpdateParticleKernel,
                          boris_particle *sp,
                          size_type s0,
                          dim3 count, dim3 strd,
                          size_type num_pic,
                          Real cmr_dt,
                          Real3 inv_dv)
{

    for (size_type x = 0; x < count.x; ++x)
        for (size_type y = 0; y < count.y; ++y)
            for (size_type z = 0; z < count.z; ++z)
            {
                for (size_type s = (s0 + x * strd.x + y * strd.y + z * strd.z) * num_pic + threadIdx.x,
                         se = s + num_pic; s < se; s += blockDim.x)
                {

                    Real rx = sp->rx[s];
                    Real ry = sp->ry[s];
                    Real rz = sp->rz[s];
                    Real vx = sp->vx[s];
                    Real vy = sp->vy[s];
                    Real vz = sp->vz[s];
                    Real f = sp->f[s];
                    Real w = sp->w[s];

                    Real ax, ay, az;
                    Real tx, ty, tz;

                    Real tt;

                    cache_gather(&ax, Ex, s0, strd.x, strd.y, strd.z, rx, ry, rz);
                    cache_gather(&ay, Ey, s0, strd.x, strd.y, strd.z, rx, ry, rz);
                    cache_gather(&az, Ez, s0, strd.x, strd.y, strd.z, rx, ry, rz);
                    cache_gather(&tx, Bx, s0, strd.x, strd.y, strd.z, rx, ry, rz);
                    cache_gather(&ty, By, s0, strd.x, strd.y, strd.z, rx, ry, rz);
                    cache_gather(&tz, Bz, s0, strd.x, strd.y, strd.z, rx, ry, rz);

                    ax *= cmr_dt;
                    ay *= cmr_dt;
                    az *= cmr_dt;

                    tx *= cmr_dt;
                    ty *= cmr_dt;
                    tz *= cmr_dt;

                    rx += vx * 0.5 * inv_dv.x;
                    ry += vy * 0.5 * inv_dv.y;
                    rz += vz * 0.5 * inv_dv.z;

                    vx += ax;
                    vy += ay;
                    vz += az;

                    Real v_x, v_y, v_z;

                    v_x = vx + (vy * tz - vz * ty);
                    v_y = vy + (vz * tx - vx * tz);
                    v_z = vz + (vx * ty - vy * tx);

                    tt = 2 / (tx * tx + ty * ty + tz * tz + 1);

                    vx += ax + (v_y * tz - v_z * ty) * tt;
                    vy += ax + (v_z * tx - v_x * tz) * tt;
                    vz += ax + (v_x * ty - v_y * tx) * tt;
                    rx += vx * 0.5 * inv_dv.x;
                    ry += vy * 0.5 * inv_dv.y;
                    rz += vz * 0.5 * inv_dv.z;


                    sp->id[s] = 0;
                    sp->rx[s] = rx;
                    sp->ry[s] = ry;
                    sp->rz[s] = rz;
                    sp->vx[s] = vx;
                    sp->vy[s] = vy;
                    sp->vz[s] = vz;
                    sp->f[s] = f;
                    sp->w[s] = w;
                }
            }

};

SP_DEVICE_DECLARE_KERNEL (spParticleBorisYeeGatherKernel,
                          boris_particle *sp,
                          size_type s0,
                          dim3 strd,
                          size_type num_pic,
                          Real cmr_dt,
                          Real3 inv_dv)
{
    Real Jx, Jy, Jz, rho;

    for (size_type s = s0 * num_pic + threadIdx.x, se = (s0 + 1) * num_pic; s < se; s += blockDim.x)
    {

        rho = 0;
        Jx = 0;
        Jy = 0;
        Jz = 0;


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
//                 if (threadId == 0)


//                atomicAdd(&g_boris_param.rho[s0], rho);
//                atomicAdd(&g_boris_param.J[0][s0], Jx);
//                atomicAdd(&g_boris_param.J[1][s0], Jy);
//                atomicAdd(&g_boris_param.J[2][s0], Jz);


    }

};


int spParticleUpdateBorisYee(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho, spField *fJ)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    boris_update_param update_param;
    update_param.max_pic = (int) spParticleGetMaxPIC(sp);
    update_param.cmr_dt = dt * spParticleGetCharge(sp) / spParticleGetMass(sp);
    size_type min[3], max[3], strides[3];
    Real inv_dv[3];
    SP_CALL(spMeshGetInvDx(m, inv_dv));
    SP_CALL(spMeshGetArrayShape(m, SP_DOMAIN_ALL, min, max, strides));
    for (int i = 0; i < 3; ++i)
    {
        inv_dv[i] *= dt;

    }

    size_type field_size = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL,
                                                     spMeshAttributeGetForm((spMeshAttribute const *) fE));
    assert(field_size < SP_CONSTANT_FIELD_SIZE);

    SP_CALL(spParticleGetAllAttributeData(sp, update_param.data));
    SP_CALL(spFieldSubArray(fRho, (void **) &update_param.rho));
    SP_CALL(spFieldSubArray(fJ, (void **) update_param.J));
    SP_CALL(spFieldSubArray((spField *) fE, (void **) update_param.E));
    SP_CALL(spFieldSubArray((spField *) fB, (void **) update_param.B));


    dim3 gridDim = {(size_type) spParallelDefaultNumOfBlocks(), 1, 1};
    dim3 blockDim = {(size_type) spParallelDefaultNumOfThreads(), 1, 1};

    for (size_type x = min[0]; x < max[0]; x += gridDim.x)
        for (size_type y = min[1]; y < max[1]; y += gridDim.y)
            for (size_type z = min[2]; z < max[2]; z += gridDim.z)
            {
                size_type s0 = x * strides[0] + y * strides[1] + z * strides[2];

                dim3 count;
                count.x = MIN(max[0] - x, gridDim.x);
                count.y = MIN(max[1] - y, gridDim.y);
                count.z = MIN(max[2] - z, gridDim.z);

                SP_CALL(spParallelMemcpyToCache((void const *) Ex,
                                                (void const *) update_param.E[0],
                                                field_size * sizeof(Real)));
                SP_CALL(spParallelMemcpyToCache((void const *) Ey,
                                                (void const *) update_param.E[1],
                                                field_size * sizeof(Real)));
                SP_CALL(spParallelMemcpyToCache((void const *) Ez,
                                                (void const *) update_param.E[2],
                                                field_size * sizeof(Real)));
                SP_CALL(spParallelMemcpyToCache((void const *) Bx,
                                                (void const *) update_param.B[0],
                                                field_size * sizeof(Real)));
                SP_CALL(spParallelMemcpyToCache((void const *) By,
                                                (void const *) update_param.B[1],
                                                field_size * sizeof(Real)));
                SP_CALL(spParallelMemcpyToCache((void const *) Bz,
                                                (void const *) update_param.B[2],
                                                field_size * sizeof(Real)));


                SP_DEVICE_CALL_KERNEL(spBorisYeeUpdateParticleKernel,
                                      (gridDim),
                                      (blockDim),
                                      (boris_particle *) sp, s0, count, sizeType2Dim3(strides),
                                      spParticleGetMaxPIC(sp),
                                      dt * spParticleGetCharge(sp) / spParticleGetMass(sp),
                                      real2Real3(inv_dv)

                );
            }

//    SP_DEVICE_CALL_KERNEL(spParticleBorisYeeGatherKernel, (gridDim), (blockDim), NULL);

    SP_CALL(spParticleSync(sp));
    SP_CALL(spFieldSync(fJ));
    SP_CALL(spFieldSync(fRho));
    return SP_SUCCESS;
}

#endif //SIMPLA_SPBORIS_DEVICE_H
