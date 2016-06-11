//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_PARTICLEINTERPOLATE_H
#define SIMPLA_PARTICLEINTERPOLATE_H

#include <math.h>
#include "ParticleInterface.h"

#include "../sp_config.h"
#include "../mesh/MeshIdHasher.h"

#define IX  1
#define IY  3
#define IZ  9


extern inline Real gather(struct point_head const *p, Real const *f, const Real *r_shift)
{
    Real r[3] = {p->r[0] - r_shift[0], p->r[1] - r_shift[1], p->r[2] - r_shift[2]};
    id_type s = (int) (r[0]) * IX + (int) (r[1]) * IY + (int) (r[2]) * IZ;

    return f[((s + IX) + IY) + IZ /* */] * r[0] * r[1] * r[2] //
           + f[(s + IX) + IY  /*     */] * r[0] * r[1] * (1.0 - r[2]) //
           + f[(s + IX) + IZ  /*     */] * r[0] * (1.0 - r[1]) * r[2] //
           + f[(s + IX)  /*          */] * r[0] * (1.0 - r[1]) * (1.0 - r[2]) //
           + f[(s + IY) + IZ  /*     */] * (1.0 - r[0]) * r[1] * r[2] //
           + f[(s + IY) /*           */] * (1.0 - r[0]) * r[1] * (1.0 - r[2]) //
           + f[s + IZ  /*            */] * (1.0 - r[0]) * (1.0 - r[1]) * r[2] //
           + f[s  /*                 */] * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
}

extern inline void gatherV(Real res[3], struct point_head const *p, Real const *f, size_t entity_type)
{
    res[0] = gather(p, f, spm_id_to_coordinates_shift_[spm_sub_index_to_id_[entity_type][0]]);
    res[1] = gather(p, f, spm_id_to_coordinates_shift_[spm_sub_index_to_id_[entity_type][1]]);
    res[2] = gather(p, f, spm_id_to_coordinates_shift_[spm_sub_index_to_id_[entity_type][2]]);
}

extern inline Real shape_factor(const struct point_head *p, Real const *r_shift)
{
    Real r[3] = {p->r[0] - r_shift[0], p->r[1] - r_shift[1], p->r[2] - r_shift[2]};

    return fabs((r[0]) * (r[1]) * (r[2]));

}

/**
 *  p->r+=inc_r
 *  @return [0,26] shift of  local cell id
 */
extern inline id_type move_point_one(struct point_head const *p, struct spPage **res, struct spPage *pool)
{
    index_type D[3] = {(index_type) (p->r[0]), (index_type) (p->r[1]), (index_type) (p->r[2])}


    p->r[0] -= D[0];
    p->r[1] -= D[1];
    p->r[2] -= D[2];

    struct point_head *p1 = spPushFront(p, &res[D[0] * IX + D[1] * IY + D[2] * IZ], pool);;
    p1->r[0] -= (Real) D[0];
    p1->r[1] -= (Real) D[1];
    p1->r[2] -= (Real) D[2];
}

void move_points(struct spPage *pg, struct spPage **res, struct spPage *pool)
{
    SP_PAGE_FOREACH(struct boris_point_s, p, pg)
    {
        move_point_one((struct point_head *) (p), res, pool);
    }
}

#endif //SIMPLA_PARTICLEINTERPOLATE_H
