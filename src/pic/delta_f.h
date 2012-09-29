/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * PIC/DeltaF.h
 *
 *  Created on: 2010-5-27
 *      Author: salmon
 */

#ifndef SRC_PIC_DELTA_F_H_
#define SRC_PIC_DELTA_F_H_
#include <string>
#include <sstream>

#include "include/simpla_defs.h"
#include "engine/object.h"
#include "engine/context.h"
#include "pic/detail/initial_random_load.h"
#include "pic/particle_pool.h"
#include "fetl/fetl.h"

namespace simpla
{
namespace pic
{
namespace delta_f
{
using namespace fetl;

struct Point_s
{
	RVec3 X, V;
	Real F;
	Real w;
};

template<typename TG>
Object::Holder InitLoadParticle(TG const & grid, const ptree & pt,
		Field<IZeroForm, Real, TG> const &n1)

{

	std::stringstream desc;
	desc << ""
			"H5T_COMPOUND {          "
			"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"X\" : "
			<< offsetof(Point_s, X)<< ";"
			"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"V\" :  "
			<< offsetof(Point_s, V) << ";"
			"   H5T_NATIVE_DOUBLE    \"F\" : "
			<< offsetof(Point_s, F) << ";"
			"   H5T_NATIVE_DOUBLE    \"W\" :  "
			<< offsetof(Point_s, w) << ";"
			"}";

	typedef ParticlePool<Point_s, TG> Pool;

	size_t particle_in_cell = pt.get<int>("pic");

	typename Pool::Holder res(new Pool(grid, sizeof(Point_s), desc.str()));

	Pool & pool = *res;

	pool.properties = pt;

	pool.resize(particle_in_cell * grid.get_numof_cell());

	RandomLoad<Point_s>(pool);

	size_t num = pool.get_numof_elements();

	double alpha = static_cast<double>(particle_in_cell);

	Real m = pool.properties.get<Real>("m");
	Real q = pool.properties.get<Real>("q");
	Real T = pool.properties.get<Real>("T");
	Real vT = sqrt(2.0 * T / m);

#pragma omp parallel for
	for (size_t s = 0; s < num; ++s)
	{
		Point_s * p = pool[s];

		p->X = p->X * (grid.xmax - grid.xmin) + grid.xmin;

		p->V = p->V * vT;

		p->F = n1(p->X) / alpha;

		p->w = 0;

	}

}

template<typename TG, typename TFE, typename TFB>
void Push(Real dt, TFE const & E1, TFB const & B0,
		ParticlePool<Point_s, TG> & pool)
{
	//   Boris' algorithm   Birdsall(1991)   p.62
	//   dv/dt = v x B
	/**
	 *  delta-f
	 *  dw/dt=(1-w) v.E/T
	 * */

	TG const & grid = pool.grid;

	Real m = pool.properties.get<Real>("m");
	Real q = pool.properties.get<Real>("q");
	Real T = pool.properties.get<Real>("T");

	Real vT = sqrt(2.0 * T / m);

	size_t num = pool.get_numof_elements();

#pragma omp parallel for
	for (size_t s = 0; s < num; ++s)
	{
		Point_s * p = pool[s];

		//FIXME there is some problem of B0
		Vec3 Bv =
		{ 0, 0, 1 };

//			Bv = (*B0)(p->X);

		Real BB = 1;	// Dot(Bv, Bv);

		Vec3 v0, v1, r0, r1;

		Vec3 Vc;

		Vc = Bv * (Dot(p->V, Bv) / BB);	// FIXME: some problem in Vector calculation

		Vec3 t, V_;
		t = Bv * (q / m * dt * 0.5);
		V_ = p->V + Cross(p->V, t);
		p->V += Cross(V_, t) * (2.0 / (Dot(t, t) + 1.0));
		Vc = Bv * (Dot(p->V, Bv) / BB);

		p->X += p->V * dt;

		for (int i = 0; i < 3; ++i)
		{
			if (grid.xmax[i] - grid.xmin[i] > 0)
			{
				if (p->X[i] > grid.xmax[i])
				{
					p->V[i] = -p->V[i];
					p->X[i] -= 2.0 * (p->X[i] - grid.xmax[i]);
				}
				if (p->X[i] < grid.xmin[i])
				{
					p->V[i] = -p->V[i];
					p->X[i] += 2.0 * (grid.xmin[i] - p->X[i]);
				}
			}
		}

		p->w = (1 - p->w) * Dot(E1(p->X), p->V) / T * dt;

	}

}

template<typename TG, typename TFE, typename TFB, typename TFJ>
void ScatterJ(ParticlePool<Point_s, TG> const & pool, TFE const & E1,
		TFB const & B0, TFJ & Js)
{
	//   Boris' algorithm   Birdsall(1991)   p.62
	//   dv/dt = v x B
	/**
	 *  delta-f
	 *  dw/dt=(1-w) v.E/T
	 * */

	Grid const & grid = pool.grid;

	Real m = pool.properties.get<Real>("m");
	Real q = pool.properties.get<Real>("q");
	Real T = pool.properties.get<Real>("T");
	Real vT = sqrt(2.0 * T / m);

	size_t num = pool.get_num_of_elements();
#pragma omp parallel
	{
		TFJ J1(grid);
		J1 = 0;
		int m = omp_get_num_threads();
		int n = omp_get_thread_num();

		for (size_t s = n * num / m; s < (n + 1) * num / m; ++s)
		{
			Point_s * p = pool[s];

			Vec3 v;
			v = p->V * (p->F * p->w);

			J1.Add(p->X, v);
		}

#pragma omp critical(PIC_ENGINE_DELTAF)
		{
			*Js += q * J1;
		}
	}
}

} // namespace delta_f
} // namespace pic
} // namespace simpla

#endif  // SRC_PIC_DELTA_F_H_
