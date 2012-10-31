/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * full_f.h
 *
 *  Created on: 2010-5-27
 *      Author: salmon
 */

#ifndef SRC_PIC_FULL_F_H_
#define SRC_PIC_FULL_F_H_
#include <string>
#include <sstream>

#include "include/simpla_defs.h"
#include "utilities/properties.h"
#include "engine/object.h"
#include "engine/modules.h"

#include "detail/initial_random_load.h"
#include "particle_pool.h"

namespace simpla
{
namespace pic
{
namespace full_f
{

struct Point_s
{
//	FullF * next;
	RVec3 X, V;
	Real F;
};

template<typename TG, typename TF>
ArrayObject::Holder InitLoadParticle(TG const & grid, const ptree & pt, TF const &n1)
{

	std::stringstream os;
	os << ""
			"H5T_COMPOUND {          "
			"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"X\" : "
			<< offsetof(Point_s, X)<< ";"
			"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"V\" :  "
			<< offsetof(Point_s, V) << ";"
			"   H5T_NATIVE_DOUBLE    \"F\" : " << offsetof(Point_s, F)
			<< ";"
			"}";

	std::string desc = os.str();

	size_t particle_in_cell = pt.get<int>("pic");

	typedef ParticlePool<Point_s, TG> Pool;

	typename Pool::Holder res(new Pool(grid, sizeof(Point_s), desc));

	Pool & pool = *res;

	pool->properties = pt;

	pool->resize(particle_in_cell * grid.get_numof_cell());

	RandomLoad<Point_s>(pool);

	size_t num = pool->get_numof_elements();

	double alpha = static_cast<double>(particle_in_cell);

	Real m = pt.get<Real>("m");
	Real q = pt.get<Real>("q");
	Real T = pt.get<Real>("T");
	Real vT = sqrt(2.0 * T / m);

#pragma omp parallel for
	for (size_t s = 0; s < num; ++s)
	{
		Point_s * p = pool[s];

		p->X = p->X * (grid.xmax - grid.xmin) + grid.xmin;

		p->V = p->V * vT;

		p->F = n1(p->X) / alpha;

	}

	return res;

}

template<typename TG, typename TFE, typename TFB>
void Push(Real dt, TFE const & E1, TFB const & B0,
		ParticlePool<Point_s, TG> & pool)
{
	TG const & grid = pool.grid;

	Real m = pool.properties.template get<Real>("m");
	Real q = pool.properties.template get<Real>("q");
	Real T = pool.properties.template get<Real>("T");
	Real vT = sqrt(2.0 * T / m);

	size_t num = pool->get_numof_elements();

	//   Boris' algorithm   Birdsall(1991)   p.62
	//   dv/dt = v x B
	/**
	 *  delta-f
	 *  dw/dt=(1-w) v.E/T
	 * */

#pragma omp parallel for
	for (size_t s = 0; s < num; ++s)
	{
		Point_s * p = pool[s];

		//FIXME there is some problem of B0
		Vec3 Bv =
		{ 0, 0, 1 };

//			Bv = (*B0)(p->X);

		Real BB = 1;		// Dot(Bv, Bv);

		Vec3 v0, v1, r0, r1;

		Vec3 E;

		E = E1(p->X);

		p->V += E * (q / m * dt * 0.5);

		Vec3 t, V_;
		t = Bv * (q / m * dt * 0.5);
		V_ = p->V + Cross(p->V, t);
		p->V += Cross(V_, t) * (2.0 / (Dot(t, t) + 1.0));

		p->V += E * (q / m * dt * 0.5);

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

	}

}

template<typename TG, typename TFE, typename TFB, typename TFJ>
void ScatterJ(ParticlePool<Point_s, TG> const & pool, TFE const & E1,
		TFB const & B0, TFJ & Js)
{

	TG const & grid = pool.grid;

	Real m = pool.properties.template get<Real>("m");
	Real q = pool.properties.template get<Real>("q");
	Real T = pool.properties.template get<Real>("T");
	Real vT = sqrt(2.0 * T / m);

	size_t num = pool->get_num_of_elements();

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
			v = p->V * (p->F);

			J1.Add(p->X, v);
		}

#pragma omp critical(PIC_ENGINE_FullF)
		{
			Js += q * J1;
		}

	}
}

template<typename TG, typename TFE, typename TFB, typename TFN>
void ScatterN(ParticlePool<Point_s, TG> const & pool, TFE const & E1,
		TFB const & B0, TFN & ns)
{

	TG const & grid = pool.grid;

	Real m = pool.properties.template get<Real>("m");
	Real q = pool.properties.template get<Real>("q");
	Real T = pool.properties.template get<Real>("T");
	Real vT = sqrt(2.0 * T / m);

	size_t num = pool->get_num_of_elements();

#pragma omp parallel
	{
		TFN n1(grid);
		n1 = 0;
		int m = omp_get_num_threads();
		int n = omp_get_thread_num();

		for (size_t s = n * num / m; s < (n + 1) * num / m; ++s)
		{
			Point_s * p = pool[s];

			n1.Add(p->X, p->F);
		}

#pragma omp critical(PIC_ENGINE_FullF)
		{
			ns += q * n1;
		}

	}
}

} // namespace full_f
} // namespace pic
} // namespace simpla

#endif  // SRC_PIC_FULL_F_H_
