/*
 * Boris.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef BORIS_H_
#define BORIS_H_
#include "../../src/sp_config.h"
#include "../../src/capi/spParticle.h"

struct boris_point_s
{
  POINT_HEAD
  Real v[3];
  Real f;
  Real w;
};
inline void
spBorisPushOne (struct boris_point_s const *p, struct boris_point_s *p_next,
				Real cmr, Real dt, Real const E[3], Real const B[3],
				const Real *inv_dx)
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
inline Real
spBorisGetRho (struct boris_point_s const *p)
{
  return p->f * p->w;
}

inline Real
spBorisGetJ (struct boris_point_s const *p, int n)
{
  return p->f * p->w * p->v[n];
}

inline Real
spBorisGetE (struct boris_point_s const *p)
{
  return 0.5 * p->f * p->w
	  * (p->v[0] * p->v[0] + p->v[1] * p->v[1] + p->v[2] * p->v[2]);
}

#endif /* BORIS_H_ */
