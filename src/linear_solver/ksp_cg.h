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
namespace linear_solver
{

template<typename TG, int IFORM, typename F1, typename F2, typename F3> //
void ksp_cg(Field<TG, IFORM, F1> const & Ax, Field<TG, IFORM, F2> const & b,
		Field<TG, IFORM, F3> & x_, size_t max_iterative_num = 1000,
		double residual = 1.0e-10)
{
//	if (!CheckEquationHasVariable(Ax, x_))
//	{
//		LOGIC_ERROR << "Unsolvable Equation!";
//	}

	typename Field<TG, IFORM, F1>::Grid const & grid = x_.grid;

	typedef typename Field<TG, IFORM, F1>::Value Value;

	Field<TG, IFORM, Value> r(grid), Ap(grid), x(grid), p(grid);

	INFORM << "KSP_CG Solver: Start";

	size_t num_of_elements = x_.grid.get_num_of_elements(IFORM);

	Value rsold, rsnew, alpha;

	r = b - Ax;
	p = r;
	x = x_;

	rsold = InnerProduct(r, r);

	for (int k = 0; k < max_iterative_num; ++k)
	{
		x_ = p;
		Ap = Ax;
		alpha = rsold / InnerProduct(p, Ap);
		x = x + alpha * (p);
		r = r - alpha * Ap;

		rsnew = InnerProduct(r, r);

		if (rsnew < residual * num_of_elements)
		{
			break;
		}
		p = r + rsnew / rsold * p;
		rsold = rsnew;
	}

	x_ = x;

	INFORM << "KSP_CG Solver: DONE! ";

// [ Residual = " 	<< rsnew / static_cast<double>(num_of_ele) << ", iterate " << k	<< " times]
}

}
// namespace linear_solver
}
// namespace simpla

#endif /* KSP_CG_H_ */
