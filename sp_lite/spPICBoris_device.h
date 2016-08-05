//
// Created by salmon on 16-8-4.
//

#ifndef SIMPLA_SPBORIS_DEVICE_H
#define SIMPLA_SPBORIS_DEVICE_H


#include "sp_lite_def.h"
#include "spPICBoris.h"
#include "spParallel_device.h"


#define ll 0
#define rr 1


static DEVICE_INLINE void
cache_gather(Real *v, Real const *f, size_type s, size_type IX, size_type IY, size_type IZ, Real rx, Real ry, Real rz)
{

    *v = *v
         + (f[s + IX + IY + IZ /*  */] * (rx - ll) * (ry - ll) * (rz - ll)
            + f[s + IX + IY /*     */] * (rx - ll) * (ry - ll) * (rr - rz)
            + f[s + IX + IZ /*     */] * (rx - ll) * (rr - ry) * (rz - ll)
            + f[s + IX /*          */] * (rx - ll) * (rr - ry) * (rr - rz)
            + f[s + IY + IZ /*     */] * (rr - rx) * (ry - ll) * (rz - ll)
            + f[s + IY /*          */] * (rr - rx) * (ry - ll) * (rr - rz)
            + f[s + IZ /*          */] * (rr - rx) * (rr - ry) * (rz - ll)
            + f[s + 0 /*           */] * (rr - rx) * (rr - ry) * (rr - rz));
}

#undef ll
#undef rr


static DEVICE_INLINE void spBoris(Real cmr_dt, Real3 mesh_inv_dv,
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
}

#endif //SIMPLA_SPBORIS_DEVICE_H
