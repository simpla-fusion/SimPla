//
// Created by salmon on 16-6-12.
//

#ifndef SIMPLA_BORIS_H
#define SIMPLA_BORIS_H

#include "../../src/particle/ParticleInterface.h"

#ifdef __cplusplus
extern "C"
{
#endif
struct boris_point_s
{
	POINT_HEAD

	double v[3];
	double f;
	double w;
};

MC_INLINE void spBorisPushOne(struct boris_point_s const *p,
		struct boris_point_s *p_next, Real cmr, double dt, Real const E[3],
		Real const B[3], const Real *inv_dx)
{

	p_next->r[0] = p->r[0] + p->v[0] * dt * 0.5 * inv_dx[0];
	p_next->r[1] = p->r[1] + p->v[1] * dt * 0.5 * inv_dx[1];
	p_next->r[2] = p->r[2] + p->v[2] * dt * 0.5 * inv_dx[2];

	Real v_[3], t[3];

	t[0] = B[0] * (cmr * dt * 0.5);
	t[1] = B[1] * (cmr * dt * 0.5);
	t[2] = B[2] * (cmr * dt * 0.5);

	p_next->v[0] = p->v[0] + E[0] * (cmr * dt * 0.5);
	p_next->v[1] = p->v[1] + E[1] * (cmr * dt * 0.5);
	p_next->v[2] = p->v[2] + E[2] * (cmr * dt * 0.5);

	v_[0] = p_next->v[0] + (p_next->v[1] * t[2] - p_next->v[2] * t[1]);
	v_[1] = p_next->v[1] + (p_next->v[2] * t[0] - p_next->v[0] * t[2]);
	v_[2] = p_next->v[2] + (p_next->v[0] * t[1] - p_next->v[1] * t[0]);

	Real tt = t[0] * t[0] + t[1] * t[1] + t[2] * t[2] + 1.0;

	p_next->v[0] += (v_[1] * t[2] - v_[2] * t[1]) * 2.0 / tt;
	p_next->v[1] += (v_[2] * t[0] - v_[0] * t[2]) * 2.0 / tt;
	p_next->v[2] += (v_[0] * t[1] - v_[1] * t[0]) * 2.0 / tt;

	p_next->v[0] += E[0] * (cmr * dt * 0.5);
	p_next->v[1] += E[1] * (cmr * dt * 0.5);
	p_next->v[2] += E[2] * (cmr * dt * 0.5);

	p_next->r[0] += p_next->v[0] * dt * 0.5 * inv_dx[0];
	p_next->r[1] += p_next->v[1] * dt * 0.5 * inv_dx[1];
	p_next->r[2] += p_next->v[2] * dt * 0.5 * inv_dx[2];

}

MC_INLINE Real spBorisGetRho(struct boris_point_s const *p)
{

	return p->f * p->w;
}

MC_INLINE Real spBorisGetJ(struct boris_point_s const *p, int n)
{
	return p->f * p->w * p->v[n];
}
;

MC_INLINE Real spBorisGetE(struct boris_point_s const *p)
{
	return 0.5 * p->f * p->w
			* (p->v[0] * p->v[0] + p->v[1] * p->v[1] + p->v[2] * p->v[2]);
}

#ifdef __cplusplus
}
;
#endif
#endif //SIMPLA_BORIS_H
