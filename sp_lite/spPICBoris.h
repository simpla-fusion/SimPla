//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_BORISYEE_H
#define SIMPLA_BORISYEE_H

#include "sp_lite_def.h"
#include "spParticle.h"
#include "spParallel.h"
#include "cuda/spParallelCUDA.h"

typedef struct boris_particle_s
{
    SP_PARTICLE_HEAD
    SP_PARTICLE_ATTR(Real, vx)
    SP_PARTICLE_ATTR(Real, vy)
    SP_PARTICLE_ATTR(Real, vz)
    SP_PARTICLE_ATTR(Real, f)
    SP_PARTICLE_ATTR(Real, w)

} boris_particle;

struct spMesh_s;
struct spField_s;

int spBorisYeeParticleCreate(spParticle **sp, struct spMesh_s const *m);

int spBorisYeeParticleInitialize(spParticle *sp, Real n0, Real T0, size_type num_pic);

int spBorisYeeParticleUpdate(spParticle *sp,
                             Real dt,
                             const struct spField_s *fE,
                             const struct spField_s *fB,
                             struct spField_s *fRho,
                             struct spField_s *fJ);


#define ll 0
#define rr 1
#define RADIUS 2
#define CACHE_EXTENT_X RADIUS*2
#define CACHE_EXTENT_Y RADIUS*2
#define CACHE_EXTENT_Z RADIUS*2
#define CACHE_SIZE (CACHE_EXTENT_X*CACHE_EXTENT_Y*CACHE_EXTENT_Z)
#define IX  1
#define IY  CACHE_EXTENT_X
#define IZ  CACHE_EXTENT_X*CACHE_EXTENT_Y

static DEVICE_INLINE void cache_gather(Real *v, Real const *f, Real rx, Real ry, Real rz)
{

    *v = *v
         + (f[IX + IY + IZ /*  */] * (rx - ll) * (ry - ll) * (rz - ll)
            + f[IX + IY /*     */] * (rx - ll) * (ry - ll) * (rr - rz)
            + f[IX + IZ /*     */] * (rx - ll) * (rr - ry) * (rz - ll)
            + f[IX /*          */] * (rx - ll) * (rr - ry) * (rr - rz)
            + f[IY + IZ /*     */] * (rr - rx) * (ry - ll) * (rz - ll)
            + f[IY /*          */] * (rr - rx) * (ry - ll) * (rr - rz)
            + f[IZ /*          */] * (rr - rx) * (rr - ry) * (rz - ll)
            + f[0 /*           */] * (rr - rx) * (rr - ry) * (rr - rz));
}

#undef ll
#undef rr
#undef IX
#undef IY
#undef IZ

static DEVICE_INLINE void spBoris(Real cmr_dt, Real3 mesh_inv_dv,
                                  const Real *Ex, const Real *Ey, const Real *Ez,
                                  const Real *Bx, const Real *By, const Real *Bz,
                                  Real *rx, Real *ry, Real *rz,
                                  Real *vx, Real *vy, Real *vz)
{
    Real ax, ay, az;
    Real tx, ty, tz;

    Real tt;

    cache_gather(&ax, Ex, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
    cache_gather(&ay, Ey, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
    cache_gather(&az, Ez, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);
    cache_gather(&tx, Bx, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
    cache_gather(&ty, By, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
    cache_gather(&tz, Bz, *rx, *ry, *rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);

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
}

#endif //SIMPLA_BORISYEE_H
