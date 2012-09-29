/* Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * PIC/GyroGauge.h
 *
 *  Created on: 2010-5-27
 *      Author: salmon
 */

#ifndef PIC_GYRO_GAUGE_H_
#define PIC_GYRO_GAUGE_H_
#include <vector>
#include <string>
#include "include/simpla_defs.h"
#include "engine/solver.h"
namespace simpla
{
namespace pic
{
namespace GyroGauge
{

struct Point_s
{

	RVec3 X, V;
	Real F;
	Real w[];

}
;

template<typename TG, typename TF>
Object::Holder InitLoadParticle(TG const & grid, const ptree & pt, TF const &n1)
{

	size_t num_of_mate = pt.get<int>("num_of_mate");

	char desc[1024];
	snprintf(desc, sizeof(desc), "H5T_COMPOUND {          "
			"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"X\" : %ul;"
			"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"V\" : %ul;"
			"   H5T_NATIVE_DOUBLE    \"F\" : %u;"
			"   H5T_ARRAY { [%d] H5T_NATIVE_DOUBLE}    \"w\" : %d;"
			"}", (offsetof(Point_s, X)),
	(offsetof(Point_s, V)),
	(offsetof(Point_s, F)),
	num_of_mate,
	(offsetof(Point_s, w)));

	size_t particle_in_cell = pt.get<int>("pic");

	size_t num = particle_in_cell * grid.get_numof_cell();

	typedef ParticlePool<Point_s, Grid> Pool;

	typename Pool::Holder res(
			new Pool(grid, sizeof(Point_s) + sizeof(Real) * num_of_mate, desc));

	Pool & pool = *res;

	pool.properties = pt;

	pool.resize(num);

	RandomLoad<Point_s>(pool);

	double alpha = static_cast<double>(particle_in_cell);

	Real m = pool.properties.get<Real>("m");
	Real q = pool.properties.get<Real>("q");
	Real T = pool.properties.get<Real>("T");
	Real vT = sqrt(2.0 * T / m);

	double alpha = static_cast<double>(particle_in_cell * num_of_mate);

#pragma omp parallel for
	for (size_t s = 0; s < num; ++s)
	{
		Point_s * p = pool[s];

		p->X = p->X * (grid.xmax - grid.xmin) + grid.xmin;

		p->V = p->V * vT;

		p->F = n1(p->X) / alpha;

		p->w = 1.0;

	}
}

template<typename TG, typename TFE, typename TFB>
void Push(Real dt, TFE const & E1, TFB const & B0,
		ParticlePool<Point_s, TG> & pool)
{

	TG const &grid = pool.grid;

	size_t num_of_mate = pool.properties.get<int>("num_of_mate");
	Real m = pool.properties.get<Real>("m");
	Real q = pool.properties.get<Real>("q");
	Real T = pool.properties.get<Real>("T");
	Real vT = sqrt(2.0 * T / m);

	Real cosdq[num_of_mate];
	Real sindq[num_of_mate];

	Real dq = TWOPI / static_cast<Real>(num_of_mate);

	for (int i = 0; i < num_of_mate; ++i)
	{
		cosdq[i] = cos(dq * i);
		sindq[i] = sin(dq * i);
	}

	size_t num_of_cells = grid.get_num_of_cell();
	size_t num = pool.get_numof_elements();

#pragma omp parallel for
	for (size_t s = 0; s < num; ++s)
	{
		Point_s & p = pool[s];

		Vec3 Bv = B0(p.X);
		Real BB = Dot(Bv, Bv);
		Real Bs = sqrt(BB);
		// --------------------------------------------------------------------
		Vec3 v0, v1, r0, r1;
		Vec3 Vc;
		Vc = (Dot(p.V, Bv) * Bv) / BB;
		v1 = Cross(p.V, Bv / Bs);
		v0 = -Cross(v1, Bv / Bs);
		r0 = -Cross(v0, Bv) / (q / m * BB);
		r1 = -Cross(v1, Bv) / (q / m * BB);

		for (int ms = 0; ms < num_of_mate; ++ms)
		{
			Vec3 v, r;
			v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
			r = (p.X + r0 * cosdq[ms] + r1 * sindq[ms]);
			p.w[ms] += 0.5 * Dot(E1(r), v) * dt;
		}
		// --------------------------------------------------------------------
		/**
		 *  delta-f
		 *  dw/dt=(1-w) v.E/T
		 * */
		// --------------------------------------------------------------------
		//   Boris' algorithm   Birdsall(1991)   p->62
		//   dv/dt = v x B
		Vec3 t, V_;
		t = Bv * q / m * dt * 0.5;
		V_ = p.V + Cross(p.V, t);
		p.V += Cross(V_, t) / (Dot(t, t) + 1.0) * 2.0;
		Vc = (Dot(p.V, Bv) * Bv) / BB;

		p.X += Vc * dt * 0.5;
		// --------------------------------------------------------------------
		v1 = Cross(p.V, Bv / Bs);
		v0 = -Cross(v1, Bv / Bs);
		r0 = -Cross(v0, Bv) / (q / m * BB);
		r1 = -Cross(v1, Bv) / (q / m * BB);
		for (int ms = 0; ms < num_of_mate; ++ms)
		{
			Vec3 v, r;
			v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
			r = (p.X + r0 * cosdq[ms] + r1 * sindq[ms]);

			p.w[ms] += 0.5 * Dot(E1(r), v) * q / T * dt;

		}
		// --------------------------------------------------------------------
		p.X += Vc * dt * 0.5;
	}
}
template<typename TG, typename TFE, typename TFB, typename TFJ>
void ScatterJ(ParticlePool<Point_s, TG> const & pool, TFE const & E1,
		TFB const & B0, TFJ & Js)
{
	TG const &grid = pool.grid;

	size_t num_of_mate = pool.properties.get<int>("num_of_mate");
	Real m = pool.properties.get<Real>("m");
	Real q = pool.properties.get<Real>("q");
	Real T = pool.properties.get<Real>("T");
	Real vT = sqrt(2.0 * T / m);

	Real cosdq[num_of_mate];
	Real sindq[num_of_mate];

	Real dq = TWOPI / static_cast<Real>(num_of_mate);

	for (int i = 0; i < num_of_mate; ++i)
	{
		cosdq[i] = cos(dq * i);
		sindq[i] = sin(dq * i);
	}

	size_t num_of_cells = grid.get_num_of_cell();
	size_t num = pool.get_numof_elements();

#pragma omp parallel
	{
		TFJ J1(grid);
		J1 = 0;
		int m = omp_get_num_threads();
		int n = omp_get_thread_num();

		for (size_t s = n * num / m; s < (n + 1) * num / m; ++s)
		{
			Point_s * p = pool[s];

			Vec3 Bv = B0(p->X);
			Real BB = Dot(Bv, Bv);
			Real Bs = sqrt(BB);
			// --------------------------------------------------------------------
			Vec3 v0, v1, r0, r1;
			Vec3 Vc;

			Vc = (Dot(p->V, Bv) * Bv) / BB;

			v1 = Cross(p->V, Bv / Bs);
			v0 = -Cross(v1, Bv / Bs);
			r0 = -Cross(v0, Bv) / (q / m * BB);
			r1 = -Cross(v1, Bv) / (q / m * BB);
			for (int ms = 0; ms < num_of_mate; ++ms)
			{
				Vec3 v, r;
				v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
				r = (p->X + r0 * cosdq[ms] + r1 * sindq[ms]);

				J1.Add(r, v * p->w[ms]);
			}
		}

#pragma omp critical(PIC_ENGINE_FullF)
		{
			Js += q * J1;
		}

	}

}

} // namespace GyroGauge
} // namespace PIC
} // namespace simpla
#endif  // PIC_GYRO_GAUGE_H_
