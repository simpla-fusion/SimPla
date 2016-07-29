//
// Created by salmon on 16-7-28.
//

#ifndef SIMPLA_FDTD_H
#define SIMPLA_FDTD_H

#include "sp_lite_def.h"
#include "../../../../../usr/local/cuda/include/host_defines.h"
#include "cuda/spParallelCUDA.h"
#include "spPhysicalConstants.h"

struct spMesh_s;
struct spField_s;

int spUpdateFieldFDTD(struct spMesh_s const *ctx,
                      Real dt,
                      const struct spField_s *fRho,
                      const struct spField_s *fJ,
                      struct spField_s *fE,
                      struct spField_s *fB);


static DEVICE_INLINE void spFDTDMaxwell(size_type s, dim3 I,
                                        Real dt, Real3 dt_inv,
                                        Real const *Rho,
                                        Real const *Jx,
                                        Real const *Jy,
                                        Real const *Jz,
                                        Real *Ex,
                                        Real *Ey,
                                        Real *Ez,
                                        Real *Bx,
                                        Real *By,
                                        Real *Bz
)
{

    Bx[s] -= ((Ez[s] - Ez[s - I.y]) * dt_inv.y - (Ey[s] - Ey[s - I.z]) * dt_inv.z) * 0.5;
    By[s] -= ((Ex[s] - Ex[s - I.z]) * dt_inv.z - (Ez[s] - Ez[s - I.x]) * dt_inv.x) * 0.5;
    Bz[s] -= ((Ey[s] - Ey[s - I.x]) * dt_inv.x - (Ex[s] - Ex[s - I.y]) * dt_inv.y) * 0.5;

    Ex[s] += ((Bz[s + I.y] - Bz[s]) * dt_inv.y - (By[s + I.z] - By[s]) * dt_inv.z) * speed_of_light2 -
             Jx[s] / epsilon0 * dt;
    Ey[s] += ((Bx[s + I.z] - Bx[s]) * dt_inv.z - (Bz[s + I.x] - Bz[s]) * dt_inv.x) * speed_of_light2 -
             Jy[s] / epsilon0 * dt;
    Ez[s] += ((By[s + I.x] - By[s]) * dt_inv.x - (Bx[s + I.y] - Bx[s]) * dt_inv.y) * speed_of_light2 -
             Jz[s] / epsilon0 * dt;

    Bx[s] -= ((Ez[s] - Ez[s - I.y]) * dt_inv.y - (Ey[s] - Ey[s - I.z]) * dt_inv.z) * 0.5;
    By[s] -= ((Ex[s] - Ex[s - I.z]) * dt_inv.z - (Ez[s] - Ez[s - I.x]) * dt_inv.x) * 0.5;
    Bz[s] -= ((Ey[s] - Ey[s - I.x]) * dt_inv.x - (Ex[s] - Ex[s - I.y]) * dt_inv.y) * 0.5;

}

#endif //SIMPLA_FDTD_H
