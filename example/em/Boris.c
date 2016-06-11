//
// Created by salmon on 16-6-11.
//

#include "Boris.h"
#include "../../src/mesh/MeshCommon.h"
#include "../../src/particle/ParticleUtility.h"
#include "../../src/particle/SmallObjPool.h"

void spBorisPushOne(struct boris_point_s *p, Real cmr, double dt, Real tE[][27], Real tB[][27],
                    const Real *inv_dx)
{


    Real E[3] = {
            gather((struct point_head const *) (p), tE[0], spm_id_to_coordinates_shift_[spm_sub_index_to_id_[EDGE][0]]),
            gather((struct point_head const *) (p), tE[1], spm_id_to_coordinates_shift_[spm_sub_index_to_id_[EDGE][1]]),
            gather((struct point_head const *) (p), tE[2], spm_id_to_coordinates_shift_[spm_sub_index_to_id_[EDGE][2]])
    };

    Real B[3] = {
            gather((struct point_head const *) (p), tB[0], spm_id_to_coordinates_shift_[spm_sub_index_to_id_[FACE][0]]),
            gather((struct point_head const *) (p), tB[1], spm_id_to_coordinates_shift_[spm_sub_index_to_id_[FACE][1]]),
            gather((struct point_head const *) (p), tB[2], spm_id_to_coordinates_shift_[spm_sub_index_to_id_[FACE][2]])
    };

    p->r[0] += p->v[0] * dt * 0.5 * inv_dx[0];
    p->r[1] += p->v[1] * dt * 0.5 * inv_dx[1];
    p->r[2] += p->v[2] * dt * 0.5 * inv_dx[2];

    Real v_[3], t[3];

    t[0] = B[0] * (cmr * dt * 0.5);
    t[1] = B[1] * (cmr * dt * 0.5);
    t[2] = B[2] * (cmr * dt * 0.5);

    p->v[0] += E[0] * (cmr * dt * 0.5);
    p->v[1] += E[1] * (cmr * dt * 0.5);
    p->v[2] += E[2] * (cmr * dt * 0.5);


    v_[0] = p->v[0] + (p->v[1] * t[2] - p->v[2] * t[1]);
    v_[1] = p->v[1] + (p->v[2] * t[0] - p->v[0] * t[2]);
    v_[2] = p->v[2] + (p->v[0] * t[1] - p->v[1] * t[0]);


    Real tt = t[0] * t[0] + t[1] * t[1] + t[2] * t[2] + 1.0;

    p->v[0] += (v_[1] * t[2] - v_[2] * t[1]) * 2.0 / tt;
    p->v[1] += (v_[2] * t[0] - v_[0] * t[2]) * 2.0 / tt;
    p->v[2] += (v_[0] * t[1] - v_[1] * t[0]) * 2.0 / tt;

    p->v[0] += E[0] * (cmr * dt * 0.5);
    p->v[1] += E[1] * (cmr * dt * 0.5);
    p->v[2] += E[2] * (cmr * dt * 0.5);

    p->r[0] += p->v[0] * dt * 0.5 * inv_dx[0];
    p->r[1] += p->v[1] * dt * 0.5 * inv_dx[1];
    p->r[2] += p->v[2] * dt * 0.5 * inv_dx[2];

}

void spBorisPushN(struct spPage *pg, Real cmr, double dt, Real const *fE, Real const *fB, const Real inv_dx[3])
{

    Real tE[3][27], tB[3][27];
    SP_PAGE_FOREACH(struct boris_point_s, p, &pg) { spBorisPushOne(p, cmr, dt, tE, tB, inv_dx); }


}

Real spBorisGather(struct spPage *pg, Real const r_shift[3])
{
    Real res = 0;
    SP_PAGE_FOREACH(struct boris_point_s, p, &pg)
    {
        res += p->f * shape_factor(p, r_shift);
    }
    return res;
}


Real spBorisGatherV(struct spPage *pg, Real const r_shift[3], int sub_index)
{
    Real res = 0;
    SP_PAGE_FOREACH(struct boris_point_s, p, &pg)
    {
        res += p->v[sub_index] * shape_factor(p, r_shift);
    }
    return res;

}

Real spBorisGatherE(struct spPage *pg, Real const r_shift[3])
{
    Real res = 0;
    SP_PAGE_FOREACH(struct boris_point_s, p, &pg)
    {
        res += 0.5 * p->f * (p->v[0] * p->v[0] + p->v[1] * p->v[1] + p->v[2] * p->v[2]) *
               shape_factor(p, r_shift);
    }
    return res;


}
