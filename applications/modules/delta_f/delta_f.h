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
#include "engine/modules.h"
#include "modules/pic/detail/initial_random_load.h"
#include "modules/pic/particle_pool.h"
#include "fetl/fetl.h"

namespace simpla
{
namespace pic
{
template<typename TV, typename TG>
class DeltaF: public Module
{
public:
	DEFINE_FIELDS(TV, TG)

	struct Point_s
	{
		RVec3 X, V;
		Real F;
		Real w;
	};
	DeltaF(Domain & d, const PTree & pt);
	virtual ~DeltaF()
	{
	}
	virtual void Eval();
private:
	Grid const & grid;

	const Real dt;
	const Real mu0;
	const Real epsilon0;
	const Real speed_of_light;
	const Real proton_mass;
	const Real elementary_charge;
	const Real eV;

	const Real m;
	const Real q;
	const Real T;
	const Real vT;

	//input
	VecZeroForm & Js;
	//output
	OneForm const & E1;
	TwoForm const & B1;

	typename ParticlePool<Point_s, TG>::Holder pool;

};

template<typename TV, typename TG>
DeltaF<TV, TG>::DeltaF(Domain & d, const PTree & pt) :
		Module(d),

		grid(d.grid<UniformRectGrid>()),

		dt(d.dt),

		mu0(d.PHYS_CONSTANTS.get<Real>("mu")),

		epsilon0(d.PHYS_CONSTANTS.get<Real>("epsilon")),

		speed_of_light(d.PHYS_CONSTANTS.get<Real>("speed_of_light")),

		proton_mass(d.PHYS_CONSTANTS.get<Real>("proton_mass")),

		elementary_charge(d.PHYS_CONSTANTS.get<Real>("elementary_charge")),

		elementary_charge(d.PHYS_CONSTANTS.get<Real>("eV")),

		m(pt.get<Real>("m") * proton_mass),

		q(pt.get<Real>("q") * elementary_charge),

		T(pt.get<Real>("T") * eV),

		vT(sqrt(2.0 * T / m)),

		B1(d.GetObject<TwoForm>("B1")),

		E1(d.GetObject<OneForm>("E1")),

		Js(d.GetObject<OneForm>("Js"))

{

	std::stringstream desc;
	desc << ""
			"H5T_COMPOUND {          "
			"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"X\" : "
			<< offsetof(Point_s, X)<< ";"
			"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"V\" :  "
			<< offsetof(Point_s, V)<< ";"
			"   H5T_NATIVE_DOUBLE    \"F\" : "
			<< offsetof(Point_s, F)<< ";"
			"   H5T_NATIVE_DOUBLE    \"W\" :  "
			<< offsetof(Point_s, w) << ";"
			"}";

	typedef ParticlePool<Point_s, TG> Pool;

	size_t particle_in_cell = pt.get<int>("pic");

	pool = d.AddObject<Pool>(pt.get<std::string>("name"),
			new Pool(grid, sizeof(Point_s), desc.str()));

	pool->properties = pt;

	pool->resize(particle_in_cell * grid.get_numof_cell());

	RandomLoad<Point_s>(pool);

	size_t num = pool->get_num_of_elements();

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

		p->w = 0;

	}

}

template<typename TV, typename TG>
void DeltaF<TV, TG>::Eval()
{
//   Boris' algorithm   Birdsall(1991)   p.62
//   dv/dt = v x B
	/**
	 *  delta-f
	 *  dw/dt=(1-w) v.E/T
	 * */

	size_t num = pool->get_num_of_elements();

#pragma omp parallel for
	for (size_t s = 0; s < num; ++s)
	{
		Point_s * p = (*pool)[s];

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

//   Boris' algorithm   Birdsall(1991)   p.62
//   dv/dt = v x B
	/**
	 *  delta-f
	 *  dw/dt=(1-w) v.E/T
	 * */


#pragma omp parallel
	{
		VecZeroForm J1(grid);
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

} // namespace pic
} // namespace simpla

#endif  // SRC_PIC_DELTA_F_H_
