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
#include "engine/object.h"
#include "engine/context.h"
#include "engine/solver.h"
#include "pic/detail/initial_random_load.h"
#include "pic/particle_pool.h"

namespace simpla
{
namespace pic
{
using namespace fetl;
template<typename, typename > struct PICEngine;

struct FullF
{
//	FullF * next;
	RVec3 X, V;
	Real F;

	static std::string get_type_desc()
	{
		std::stringstream stream;
		stream << ""
				"H5T_COMPOUND {          "
				"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"X\" : "
				<< offsetof(FullF, X) << ";"
						"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"V\" :  "
				<< offsetof(FullF, V) << ";"
						"   H5T_NATIVE_DOUBLE    \"F\" : " << offsetof(FullF, F)
				<< ";"
						"}";

		return (stream.str());
	}

	static size_t get_size_in_bytes()
	{
		return (sizeof(FullF));
	}
};
template<typename TG>
class PICEngine<FullF, TG>
{
public:

	typedef FullF Point_s;

	typedef TG Grid;

	typedef ParticlePool<Point_s, Grid> Pool;

	typedef PICEngine<Point_s, Grid> ThisType;

	DEFINE_FIELDS(Real,TG);

	TR1::shared_ptr<Context> ctx_;

	Grid const & grid;

	typedef TR1::shared_ptr<ThisType> Holder;

	PICEngine(ThisType const & rhs) :
			Solver(rhs), Pool(rhs), ctx_(rhs.ctx_), grid(rhs.grid), name_(
					rhs.name_), m_(rhs.m_), q_(rhs.q_), T_(rhs.T_), vT_(rhs.vT_)
	{
	}

	PICEngine(TR1::shared_ptr<Context> ctx) :
			ctx_(ctx), grid(ctx->getGrid<TG>()), name_("UnNamed"), //
			m_(1.0), q_(1.0), T_(1.0), vT_(sqrt(2.0 * T_ / m_)), pic_(100)
	{
	}
	static Holder Create(TR1::shared_ptr<Context> ctx)
	{
		return (Holder(new ThisType(ctx)));
	}

	virtual ~PICEngine()
	{
	}

	void set_property(std::string const & name, Real m, Real q, Real T0,
			size_t particle_in_cell)
	{
		name_ = name;
		m_ = m;
		q_ = q;
		T_ = T0;
		vT_ = sqrt(2.0 * T_ / m_);
		pic_ = particle_in_cell;
	}

	void PreProcess()
	{
		pool_ = ctx_->template GetObject<Pool>(name_);

		if (pool_->Empty())
		{
			pool_->resize(pic_ * grid.get_num_of_cell());
			RandomLoad<FullF>(pool_);
		}

		ZeroForm &n1 = *(ctx_->template GetObject<ZeroForm>("n1"));

		size_t num = pool_->get_num_of_elements();

		double alpha = static_cast<double>(pic_);

#pragma omp parallel for
		for (size_t s = 0; s < num; ++s)
		{
			Point_s * p = (*pool_)[s];

			p->X = p->X * (grid.xmax - grid.xmin) + grid.xmin;

			p->V = p->V * vT_;

			p->F = n1(p->X) / alpha;

		}

		B0 = ctx_->template GetObject<TwoForm>("B0");
		E1 = ctx_->template GetObject<OneForm>("E1");
//		B1 =ctx_->template GetObject<TwoForm>("B1");
		Js = ctx_->template GetObject<OneForm>("J1");
		ns = ctx_->template GetObject<ZeroForm>("n1");

	}
	void Process()
	{
		Push();
		pool_->Sort();
		Scatter();
	}

	void PostProcess()
	{
	}

	void Push()
	{
		//   Boris' algorithm   Birdsall(1991)   p.62
		//   dv/dt = v x B
		/**
		 *  delta-f
		 *  dw/dt=(1-w) v.E/T
		 * */

		Real dt = grid.dt;

		size_t num = pool_->get_num_of_elements();
#pragma omp parallel for
		for (size_t s = 0; s < num; ++s)
		{
			Point_s * p = (*pool_)[s];

			//FIXME there is some problem of B0
			Vec3 Bv =
			{ 0, 0, 1 };

//			Bv = (*B0)(p->X);

			Real BB = 1; // Dot(Bv, Bv);

			Vec3 v0, v1, r0, r1;

			Vec3 E;

			E = (*E1)(p->X);

			p->V += E * (q_ / m_ * dt * 0.5);

			Vec3 t, V_;
			t = Bv * (q_ / m_ * dt * 0.5);
			V_ = p->V + Cross(p->V, t);
			p->V += Cross(V_, t) * (2.0 / (Dot(t, t) + 1.0));

			p->V += E * (q_ / m_ * dt * 0.5);

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

	void Scatter()
	{
		//   Boris' algorithm   Birdsall(1991)   p.62
		//   dv/dt = v x B
		/**
		 *  delta-f
		 *  dw/dt=(1-w) v.E/T
		 * */

		size_t num = pool_->get_num_of_elements();
#pragma omp parallel
		{
			OneForm J1(grid);
			ZeroForm n1(grid);
			n1 = 0;
			J1 = 0;
			int m = omp_get_num_threads();
			int n = omp_get_thread_num();

			for (size_t s = n * num / m; s < (n + 1) * num / m; ++s)
			{
				Point_s * p = (*pool_)[s];

				Vec3 v;
				v = p->V * (p->F);

				J1.Add(p->X, v);
				n1.Add(p->X, p->F);
			}

#pragma omp critical(PIC_ENGINE_FullF)
			{
				*Js += q_ * J1;
				*ns += q_ * n1;
			}

		}
	}

private:
	TR1::shared_ptr<Pool> pool_;
	std::string name_;
	size_t pic_;
	Real m_, q_, T_;
	Real vT_;

	TR1::shared_ptr<TwoForm> B0;
	TR1::shared_ptr<OneForm> E1;
	TR1::shared_ptr<TwoForm> B1;
	TR1::shared_ptr<OneForm> Js;
	TR1::shared_ptr<ZeroForm> ns;

}
;
} // namespace pic
} // namespace simpla

#endif  // SRC_PIC_FULL_F_H_
