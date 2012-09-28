/*
 * ksp_cg.h
 *
 *  Created on: 2012-3-30
 *      Author: salmon
 */

#ifndef KSP_CG_H_
#define KSP_CG_H_
#include "include/simpla_defs.h"
#include "fetl/fetl.h"
#include "utilities/log.h"
#include "engine/solver.h"
#include <iostream>

namespace simpla
{
using namespace simpla::fetl;
namespace linear_solver
{

template<typename TEqn, typename TF>
struct KSP_CG: public Solver
{
	KSP_CG(TEqn const & Ax_b, TF & x, size_t max_iterative_num = 1000,
			double residual = 1.0e-10) :
			flag_done_(false), max_iterative_num_(max_iterative_num),

			residual_(residual),

			num_of_ele_(x.grid.get_num_of_elements(x.IForm)),

			Ax_b_(Ax_b), x_(x), r(Ax_b.grid), b(Ax_b.grid), Ap(Ax_b.grid), p(
					x.grid), x0(x.grid)
	{
		if (!CheckEquationHasVariable(Ax_b_, x))
		{
			LOGIC_ERROR<< "Unsolvable Equation!";
		}
	}
	~KSP_CG()
	{
		if (!flag_done_)
		{
			Process();
		}
	}

	void PreProcess()
	{
	}

	void Process()
	{
		flag_done_ = true;

		INFORM<<"KSP_CG Solver: Start";

		typename TEqn::ValueType rsold, rsnew, alpha;

		r = -Ax_b_;
		p = r;
		x0 = x_;
		x_ = 0;
		b = -Ax_b_;
		rsold = InnerProduct(r, r);

		size_t k = 0;
		while (k < max_iterative_num_)
		{
			x_ = p;
			Ap = Ax_b_ + b;
			alpha = rsold / InnerProduct(p, Ap);
			x0 = x0 + alpha * (p);
			r = r - alpha * Ap;
			rsnew = InnerProduct(r, r);

			if (sqrt(rsnew / static_cast<double>(num_of_ele_)) < residual_)
			{
				break;
			}
			p = r + rsnew / rsold * p;
			rsold = rsnew;

			++k;
		}
		x_ = x0;

		INFORM << "KSP_CG Solver: DONE! [ Residual = "
				<< rsnew / static_cast<double>(num_of_ele_) << ", iterate " << k
				<< " times]";
	}

	void PostProcess()
	{
	}

private:
	bool flag_done_;
	size_t max_iterative_num_;
	double residual_;
	size_t num_of_ele_;
	TEqn const & Ax_b_;
	TF & x_;

	typename FieldTraits<TEqn>::FieldType r;
	typename FieldTraits<TEqn>::FieldType b;
	typename FieldTraits<TEqn>::FieldType Ap;
	typename FieldTraits<TF>::FieldType p;
	typename FieldTraits<TF>::FieldType x0;

};
template<typename TEqn, typename TF>
inline KSP_CG<TEqn, TF> ksp_cg_solver(TEqn const & Ax_b, TF & x,
		size_t max_iterative_num = 1000, double residual = 1.0e-10)
{
	return (KSP_CG<TEqn, TF>(Ax_b, x, max_iterative_num, residual));
}

template<typename TAx, typename Tb, typename Tx>
struct KSP_CG2: public Solver
{

	KSP_CG2(TAx const & Ax, Tb const & b, Tx & x, size_t max_iterative_num =
			1000, double residual = 1.0e-10) :
			flag_done_(false), max_iterative_num_(max_iterative_num),

			residual_(residual),

			num_of_ele_(x.grid.get_num_of_elements(x.IForm)),

			Ax_(Ax), b_(b), x_(x), r(Ax.grid), Ap(Ax.grid), p(x.grid), x0(
					x.grid)
	{
		if (!CheckEquationHasVariable(Ax_, x)
				|| CheckEquationHasVariable(b_, x))
		{
			LOGIC_ERROR<< "Unsolvable Equation!";
		}
	}
	~KSP_CG2()
	{
		if (!flag_done_)
		{
			Process();
		}
	}

	void PreProcess()
	{
	}

	void Process()
	{
		flag_done_ = true;

		INFORM<<"KSP_CG Solver: Start";

		typename TAx::ValueType rsold, rsnew, alpha;

		r = b_ - Ax_;
		p = r;
		x0 = x_;
		x_ = 0;
		rsold = InnerProduct(r, r);

		size_t k = 0;
		while (k < max_iterative_num_)
		{
			x_ = p;
			Ap = Ax_;
			alpha = rsold / InnerProduct(p, Ap);
			x0 = x0 + alpha * (p);
			r = r - alpha * Ap;
			rsnew = InnerProduct(r, r);

			if (sqrt(rsnew / static_cast<double>(num_of_ele_)) < residual_)
			{
				break;
			}
			p = r + rsnew / rsold * p;
			rsold = rsnew;

			++k;
		}
		x_ = x0;

		INFORM << "KSP_CG Solver: DONE! [ Residual = "
				<< rsnew / static_cast<double>(num_of_ele_) << ", iterate " << k
				<< " times]";
	}

	void PostProcess()
	{
	}

private:
	bool flag_done_;
	size_t max_iterative_num_;
	double residual_;
	size_t num_of_ele_;
	TAx const & Ax_;
	Tb const & b_;
	Tx & x_;

	typename FieldTraits<TAx>::FieldType r;
	typename FieldTraits<TAx>::FieldType Ap;
	typename FieldTraits<Tx>::FieldType p;
	typename FieldTraits<Tx>::FieldType x0;

};

template<typename TAx, typename Tb, typename Tx>
inline KSP_CG2<TAx, Tb, Tx> ksp_cg_solver(TAx const & Ax, Tb const &b, Tx & x,
		size_t max_iterative_num = 1000, double residual = 1.0e-10)
{
	return (KSP_CG2<TAx, Tb, Tx>(Ax, b, x, max_iterative_num, residual));
}
template<typename TAx, typename Tb, typename Tx>
inline KSP_CG2<TAx, Tb, Tx> ksp_cg_solver(FieldEquation<TAx, Tb> const & eqn
		, Tx & x, size_t max_iterative_num = 1000, double residual = 1.0e-10)
{
	return (KSP_CG2<TAx, Tb, Tx>(eqn.lhs_, eqn.rhs_, x, max_iterative_num,
			residual));
}
} // namespace linear_solver
} // namespace simpla

#endif /* KSP_CG_H_ */
