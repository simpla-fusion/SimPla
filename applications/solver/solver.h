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

#include "electromagnetic/pml.h"
#include "electromagnetic/cold_fluid.h"

namespace simpla
{

template<typename TCfg, typename TM, typename TE, typename TB>
void CreateEMFieldSolver(TCfg const & cfg, TM const & mesh,
        std::function<void(Real, TE const &, TB const &, TE*)> *solverE,
        std::function<void(Real, TE const &, TB const &, TB*)> *solverB)
{

	*solverB = [](Real dt,TE const &E, TB const &B, TB* dB)
	{
		LOG_CMD(*dB -= Curl(E)*dt);
	};

	*solverE = [](Real dt,TE const &E, TB const &B, TE* dE)
	{
		LOG_CMD(*dE += Curl(B)*dt);
	};

	if (cfg)
	{

		if (cfg["ColdFluid"])
		{
			auto solver = std::shared_ptr<ColdFluidEM<TM> >(new ColdFluidEM<TM>(mesh));

			solver->Load(cfg["ColdFluid"]);

			*solverE = std::bind(&ColdFluidEM<TM>::NextTimeStepE, solver);
		}

		if (cfg["PML"])
		{
			auto solver = std::shared_ptr<ColdFluidEM<TM> >(new PML<TM>(mesh));

			solver->Load(cfg["PML"]);

			*solverE = std::bind(&PML<TM>::NextTimeStepE, solver);

			*solverB = std::bind(&PML<TM>::NextTimeStepB, solver);

		}
	}
}

}
// namespace simpla

#endif /* SOLVER_H_ */
