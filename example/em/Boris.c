//
// Created by salmon on 16-6-11.
//

#include "Boris.h"
#include "../../src/mesh/MeshCommon.h"
#include "../../src/particle/ParticleUtility.h"
#include "../../src/particle/SmallObjPool.h"

void spBorisPush(struct boris_point_s *p, Real cmr, double dt, Real const *fE, Real const *fB,
                 index_type const *i_lower, index_type const *i_upper, const Real inv_dx[3])
{
    Real E[3], B[3];
    gatherV(E, (struct point_head const *) (p), fE, EDGE, i_lower, i_upper);
    gatherV(B, (struct point_head const *) (p), fB, FACE, i_lower, i_upper);


    Real inc_x[3] = {p->v[0] * dt * 0.5, p->v[1] * dt * 0.5, p->v[2] * dt * 0.5};

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

    inc_x[0] += p->v[0] * dt * 0.5;
    inc_x[1] += p->v[1] * dt * 0.5;
    inc_x[2] += p->v[2] * dt * 0.5;

    move_particle((struct point_head *) (p), inc_x, inv_dx);
}

void spBorisPushN(struct spPage *pg, Real cmr, double dt, Real const *fE, Real const *fB,
                  index_type const *i_lower, index_type const *i_upper, const Real inv_dx[3])
{
    SP_PAGE_FOREACH(struct boris_point_s, p, &pg) { spBorisPush(p, cmr, dt, fE, fB, i_lower, i_upper, inv_dx); }
}

Real spBorisGather(struct spPage *pg, Real const r_shift[3])
{
    Real res = 0;
    SP_PAGE_FOREACH(struct boris_point_s, p, &pg)
    {
        res += p->f * shape_factor(p->r, r_shift);
    }
    return res;
}


Real spBorisGatherV(struct spPage *pg, Real const r_shift[3], int sub_index)
{
    Real res = 0;
    SP_PAGE_FOREACH(struct boris_point_s, p, &pg)
    {
        res += p->v[sub_index] * shape_factor(p->r, r_shift);
    }
    return res;

}

Real spBorisGatherE(struct spPage *pg, Real const r_shift[3])
{
    Real res = 0;
    SP_PAGE_FOREACH(struct boris_point_s, p, &pg)
    {
        res += 0.5 * p->f * (p->v[0] * p->v[0] + p->v[1] * p->v[1] + p->v[2] * p->v[2]) * shape_factor(p->r, r_shift);
    }
    return res;


}
