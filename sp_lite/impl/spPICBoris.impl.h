//
// Created by salmon on 16-8-4.
//

#ifndef SIMPLA_SPBORIS_DEVICE_H
#define SIMPLA_SPBORIS_DEVICE_H


#include "../sp_lite_def.h"
#include "../spPICBoris.h"
#include "spParallel.impl.h"
#include "../spRandom.h"
#include <math.h>

#define ll 0
#define rr 1
#define   IX   1
#define   IY   3
#define   IZ   9

INLINE DEVICE void
cache_gather(Real *v, Real const *f, size_type s_c, Real rx, Real ry, Real rz)
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

INLINE DEVICE void
cache_scatter(Real v, Real *f, Real rx, Real ry, Real rz, size_type s_c)
{
    atomicAdd(&f[s_c + IX + IY + IZ /**/], (v * (rx - ll) * (ry - ll) * (rz - ll)));
    atomicAdd(&f[s_c + IX + IY /*     */], (v * (rx - ll) * (ry - ll) * (rr - rz)));
    atomicAdd(&f[s_c + IX + IZ /*     */], (v * (rx - ll) * (rr - ry) * (rz - ll)));
    atomicAdd(&f[s_c + IX /*          */], (v * (rx - ll) * (rr - ry) * (rr - rz)));
    atomicAdd(&f[s_c + IY + IZ /*     */], (v * (rr - rx) * (ry - ll) * (rz - ll)));
    atomicAdd(&f[s_c + IY /*          */], (v * (rr - rx) * (ry - ll) * (rr - rz)));
    atomicAdd(&f[s_c + IZ /*          */], (v * (rr - rx) * (rr - ry) * (rz - ll)));
    atomicAdd(&f[s_c + 0 /*           */], (v * (rr - rx) * (rr - ry) * (rr - rz)));
}


#undef ll
#undef rr
#undef   IX
#undef   IY
#undef   IZ
#undef  s_c

INLINE DEVICE void spBoris(Real cmr_dt, Real3 mesh_inv_dv,
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

    cache_gather(&ax, Ex, 0, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
    cache_gather(&ay, Ey, 0, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
    cache_gather(&az, Ez, 0, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);
    cache_gather(&tx, Bx, 0, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
    cache_gather(&ty, By, 0, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
    cache_gather(&tz, Bz, 0, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);

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

    int dist_type[6] = {SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_NORMAL, SP_RAND_NORMAL, SP_RAND_NORMAL};

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

#endif //SIMPLA_SPBORIS_DEVICE_H
