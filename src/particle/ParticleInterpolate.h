//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_PARTICLEINTERPOLATE_H
#define SIMPLA_PARTICLEINTERPOLATE_H

#include <math.h>
#include "ParticleEngine.h"

#include "../sp_config.h"
#include "../mesh/MeshIdHasher.h"

inline double gather(double *f, point_head const *p, id_type shift, index_type const i_lower[],
                     index_type const i_upper[])
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

inline double scatter(point_head const *p, id_type shift)
{
    return fabs((d[0] - r[0]) * (d[1] - r[1]) * (d[2] - r[2]));


}

#endif //SIMPLA_PARTICLEINTERPOLATE_H
