/*
 * solver.h
 *
 *  Created on: 2014年1月1日
 *      Author: salmon
 */

#ifndef SOLVER_H_
#define SOLVER_H_

#include <memory>
#include <string>

#include "pml.h"
#include "cold_fluid.h"

namespace simpla
{

template<typename TDict, typename TM, typename TE, typename TB>
void CreateEMSolver(TDict const & dict, TM const & mesh,
        std::function<void(Real, TE const &, TB const &, TE*)> *solverE,
        std::function<void(Real, TE const &, TB const &, TB*)> *solverB)
{
	using namespace std::placeholders;

	if (!dict)
		return;

	if (dict["ColdFluid"])
	{
		auto solver = std::shared_ptr<ColdFluidEM<TM> >(new ColdFluidEM<TM>(mesh));

		solver->Load(dict["ColdFluid"]);

		*solverE = std::bind(&ColdFluidEM<TM>::template NextTimeStepE<TE, TB>, solver, _1, _2, _3, _4);
	}

	if (dict["PML"])
	{
		auto solver = std::shared_ptr<PML<TM> >(new PML<TM>(mesh));

		solver->Load(dict["PML"]);

		*solverE = std::bind(&PML<TM>::NextTimeStepE, solver, _1, _2, _3, _4);

		*solverB = std::bind(&PML<TM>::NextTimeStepB, solver, _1, _2, _3, _4);

	}

	LOGGER << "Load electromagnetic field solver" << DONE;

}

}
// namespace simpla

#endif /* SOLVER_H_ */
