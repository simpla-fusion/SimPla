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

struct GyroGauge
{
	typedef GyroGauge ThisType;

	ThisType * next;
	RVec3 X, V;
	Real F;
	Real w[];

	static std::string get_value_type_desc(int num_of_mate)
	{
		char cbuff[1024];
		snprintf(cbuff, sizeof(cbuff), "H5T_COMPOUND {          "
				"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"X\" : %ul;"
				"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"V\" : %ul;"
				"   H5T_NATIVE_DOUBLE    \"F\" : %u;"
				"   H5T_ARRAY { [%d] H5T_NATIVE_DOUBLE}    \"w\" : %d;"
				"}",

		(offsetof(ThisType, X)),

		(offsetof(ThisType, V)),

		(offsetof(ThisType, F)),

		num_of_mate,

		(offsetof(ThisType, w)));

		return std::string(cbuff);
	}
	static size_t get_value_size_in_bytes(int num_of_mate = 4)
	{
		return (sizeof(GyroGauge) + sizeof(Real) * num_of_mate);
	}

};

template<typename > class PICEngine;

template<typename TG, template<typename, typename > class TPool>
class PICEngine<TPool<GyroGauge, TG> > : public Solver
{
public:

	typedef TG Grid;

	typedef GyroGauge Point_s;

	typedef TPool<Point_s, Grid> Pool;

	typedef PICEngine<Pool> ThisType;

	Grid const &grid;

	PICEngine(Grid const &pgrid) :
			grid(pgrid), num_of_mate_(4), //
			m_(1.0), q_(1.0), T_(1.0), vT_(sqrt(2.0 * T_ / m_))
	{
	}
	~PICEngine()
	{
	}

	void set_property(Real m, Real q, Real T0, int num_of_mate)
	{
		num_of_mate_ = num_of_mate;
		m_ = m;
		q_ = q;
		T_ = T0;
		vT_ = sqrt(2.0 * T_ / m_);

		cosdq.resize(num_of_mate_);
		sindq.resize(num_of_mate_);

		Real dq = TWOPI / static_cast<Real>(num_of_mate_);

		for (int i = 0; i < num_of_mate_; ++i)
		{
			cosdq[i] = cos(dq * i);
			sindq[i] = sin(dq * i);
		}

	}

	template<typename TContexHolder>
	inline void Initialize(TContexHolder ctx, int pic)
	{

		typename Pool::Holder(
				Pool(grid, Point_s::get_value_size_in_bytes(num_of_mate_),
						Point_s::get_value_type_desc(num_of_mate_))).swap(
				pool_);

		pool_.Initialize(pic);

		size_t num_of_cells = grid.get_num_of_cell();

		Vec3 w =
		{ 1, 1, 1 };

#pragma omp parallel for
		for (size_t s = 0; s < num_of_cells; ++s)
		{

			RVec3 X0 = grid.get_cell_center(s);

			Grid sgrid = grid.SubGrid(X0, w);

			ZeroForm n1(sgrid);

			n1 = *(ctx->find(n1)->second);

			for (Point_s * p = (*pool_)[s]; p != NULL; p = p->next)
			{
				p->X = p->X * Pool::grid.dx + X0;
				p->V = p->V * vT_;
				p->F = n1(p->X) * w / static_cast<Real>(num_of_mate_);

			}
		}

	}

	template<typename TContexHolder>
	void Push(TContexHolder ctx)
	{
		size_t num_of_cells = grid.get_num_of_cell();

		Vec3 w =
		{ 1, 1, 1 };
		Real dt = grid.dt;
#pragma omp parallel for
		for (size_t s = 0; s < num_of_cells; ++s)
		{

			RVec3 X0 = grid.get_cell_center(s);

			Grid sgrid = grid.SubGrid(X0, w);

			VecZeroForm B0(sgrid);
			B0 = *(ctx->find("B0")->second);
			OneForm E1(sgrid);
			E1 = *(ctx->find("E1")->second);
			TwoForm B1(sgrid);
			B1 = *(ctx->find("B1")->second);

			for (Point_s * p = (*pool_)[s]; p != NULL; p = p->next)
			{

				Vec3 Bv = B0(p->X);
				Real BB = Dot(Bv, Bv);
				Real Bs = sqrt(BB);
				// --------------------------------------------------------------------
				Vec3 v0, v1, r0, r1;
				Vec3 Vc;
				Vc = (Dot(p->V, Bv) * Bv) / BB;
				v1 = Cross(p->V, Bv / Bs);
				v0 = -Cross(v1, Bv / Bs);
				r0 = -Cross(v0, Bv) / (q_ / m_ * BB);
				r1 = -Cross(v1, Bv) / (q_ / m_ * BB);

				for (int ms = 0; ms < num_of_mate_; ++ms)
				{
					Vec3 v, r;
					v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
					r = (p->X + r0 * cosdq[ms] + r1 * sindq[ms]);
					p->w[ms] += 0.5 * Dot(E1(r), v) * dt;
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
				t = Bv * q_ / m_ * dt * 0.5;
				V_ = p->V + Cross(p->V, t);
				p->V += Cross(V_, t) / (Dot(t, t) + 1.0) * 2.0;
				Vc = (Dot(p->V, Bv) * Bv) / BB;

				p->X += Vc * dt * 0.5;
				// --------------------------------------------------------------------
				v1 = Cross(p->V, Bv / Bs);
				v0 = -Cross(v1, Bv / Bs);
				r0 = -Cross(v0, Bv) / (q_ / m_ * BB);
				r1 = -Cross(v1, Bv) / (q_ / m_ * BB);
				for (int ms = 0; ms < num_of_mate_; ++ms)
				{
					Vec3 v, r;
					v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
					r = (p->X + r0 * cosdq[ms] + r1 * sindq[ms]);

					p->w[ms] += 0.5 * Dot(E1(r), v) * q_ / T_ * dt;

				}
				// --------------------------------------------------------------------
				p->X += Vc * dt * 0.5;
			}
		}
	}

	template<typename TContexHolder>
	void Scatter(TContexHolder ctx)
	{
		size_t num_of_cells = grid.get_num_of_cell();

		Vec3 w =
		{ 1, 1, 1 };

#pragma omp parallel for
		for (size_t s = 0; s < num_of_cells; ++s)
		{

			RVec3 X0 = grid.get_cell_center(s);

			Grid sgrid = grid.SubGrid(X0, w);

			VecZeroForm B0(sgrid);
			B0 = *(ctx->find("B0")->second);
			VecZeroForm J1(sgrid);
			J1 = 0.0;
			ZeroForm n1(sgrid);
			n1 = 0.0;

			for (Point_s * p = (*pool_)[s]; p != NULL; p = p->next)
			{

				Vec3 Bv = B0(p->X);
				Real BB = Dot(Bv, Bv);
				Real Bs = sqrt(BB);
				// --------------------------------------------------------------------
				Vec3 v0, v1, r0, r1;
				Vec3 Vc;

				Vc = (Dot(p->V, Bv) * Bv) / BB;

				v1 = Cross(p->V, Bv / Bs);
				v0 = -Cross(v1, Bv / Bs);
				r0 = -Cross(v0, Bv) / (q_ / m_ * BB);
				r1 = -Cross(v1, Bv) / (q_ / m_ * BB);
				for (int ms = 0; ms < num_of_mate_; ++ms)
				{
					Vec3 v, r;
					v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
					r = (p->X + r0 * cosdq[ms] + r1 * sindq[ms]);

					J1.Add(r, v);
					n1.Add(r, p->w[ms]);
				}

			}
#pragma omp critical(PICENGINE_DELTAF)
			{
				*(ctx->find("J1")->second) += q_ * J1;
				*(ctx->find("n1")->second) += q_ * n1;
			}
		}
	}

private:
	typename Pool::Holder pool_;

	Real m_, q_, T_;
	Real vT_;

	int num_of_mate_;
	std::vector<Real> cosdq, sindq;
};

} // namespace PIC
} // namespace simpla
#endif  // PIC_GYRO_GAUGE_H_
