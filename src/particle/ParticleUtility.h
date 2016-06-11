//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_PARTICLEINTERPOLATE_H
#define SIMPLA_PARTICLEINTERPOLATE_H

#include <math.h>
#include "ParticleCommon.h"

#include "../sp_config.h"
#include "../mesh/MeshIdHasher.h"

extern inline double gather(struct point_head const *p, double const *f, id_type shift, const index_type i_lower[],
                            const index_type i_upper[])
{
    double const *r = p->r;
    id_type s = p->_cell - shift;

    auto X = (_DI) << 1;
    auto Y = (_DJ) << 1;
    auto Z = (_DK) << 1;
    return f[sp_hash(((s + X) + Y) + Z,/*  */i_lower, i_upper)] * r[0] * r[1] * r[2] //
           + f[sp_hash((s + X) + Y, /*     */i_lower, i_upper)] * r[0] * r[1] * (1.0 - r[2]) //
           + f[sp_hash((s + X) + Z, /*     */i_lower, i_upper)] * r[0] * (1.0 - r[1]) * r[2] //
           + f[sp_hash((s + X), /*         */i_lower, i_upper)] * r[0] * (1.0 - r[1]) * (1.0 - r[2]) //
           + f[sp_hash((s + Y) + Z, /*     */i_lower, i_upper)] * (1.0 - r[0]) * r[1] * r[2] //
           + f[sp_hash((s + Y),/*          */i_lower, i_upper)] * (1.0 - r[0]) * r[1] * (1.0 - r[2]) //
           + f[sp_hash(s + Z, /*           */i_lower, i_upper)] * (1.0 - r[0]) * (1.0 - r[1]) * r[2] //
           + f[sp_hash(s, /*               */i_lower, i_upper)] * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
}

extern inline void gatherV(Real res[3], struct point_head const *p, double const *f, size_t entity_type,
                           const index_type i_lower[], const index_type i_upper[])
{
    res[0] = gather(p, f, spm_id_to_shift_[spm_sub_index_to_id_[1][0]], i_lower, i_upper);
    res[1] = gather(p, f, spm_id_to_shift_[spm_sub_index_to_id_[1][1]], i_lower, i_upper);
    res[2] = gather(p, f, spm_id_to_shift_[spm_sub_index_to_id_[1][2]], i_lower, i_upper);
}

extern inline Real shape_factor(Real const r0[3], Real const r1[3])
{
    return fabs((r1[0] - r0[0]) * (r1[1] - r0[1]) * (r1[2] - r0[2]));

}

extern inline void move_particle(struct point_head *p, Real const inc_x[3], Real const inv_dx[3])
{
    index_type D[3];
    p->r[0] += inc_x[0] * inv_dx[0];
    p->r[1] += inc_x[1] * inv_dx[1];
    p->r[2] += inc_x[2] * inv_dx[2];

    D[0] = (index_type) (p->r[0]);
    p->r[0] -= D[0];
    D[1] = (index_type) (p->r[1]);
    p->r[1] -= D[1];
    D[2] = (index_type) (p->r[2]);
    p->r[2] -= D[2];
    p->_cell += sp_pack(D[0], D[1], D[2]);
}

#endif //SIMPLA_PARTICLEINTERPOLATE_H
