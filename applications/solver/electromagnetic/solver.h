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

template<typename TDict, typename TM, typename TE, typename TB, typename ...Args>
std::string CreateEMSolver(TDict const & dict, TM const & mesh,
        std::function<void(Real, TE const &, TB const &, TE*)> *solveE,
        std::function<void(Real, TE const &, TB const &, TB*)> *solveB, Args const & ... args)
{

	std::ostringstream os;

	LOGGER << "Load Electromagnetic fields solver";

	using namespace std::placeholders;

	DEFINE_PHYSICAL_CONST(mesh.constants());

	Real ic2 = 1.0 / (mu0 * epsilon0);

	std::function<void(Real, TE const &, TB const &, TE*)> sE = //
	        [mu0 , epsilon0](Real dt, TE const & , TB const & B, TE* pdE)
	        {
		        auto & dE=*pdE;
		        LOG_CMD(dE += Curl(B)/(mu0 * epsilon0) *dt);
	        };

	std::function<void(Real, TE const &, TB const &, TB*)> sB = //
	        [](Real dt, TE const & E, TB const &, TB* pdB)
	        {
		        auto & dB=*pdB;
		        LOG_CMD( dB -= Curl(E)*dt);
	        };

	if (dict["PML"])
	{
		auto solver = std::shared_ptr<PML<TM> >(new PML<TM>(mesh));

		solver->Load(dict["PML"]);

		solver->Print(os);

		sE = std::bind(&PML<TM>::NextTimeStepE, solver, _1, _2, _3, _4);

		sB = std::bind(&PML<TM>::NextTimeStepB, solver, _1, _2, _3, _4);

	}

	*solveB = sB;
	*solveE = sE;

	if (dict["ColdFluid"])
	{
		auto solver = std::shared_ptr<ColdFluidEM<TM> >(new ColdFluidEM<TM>(mesh));

		solver->Load(dict["ColdFluid"], std::forward<Args const &>(args)...);

		solver->Print(os);

		*solveE = [=](Real dt, TE const & E , TB const & B, TE* pdE)
		{
			sE(dt,E,B,pdE);
			solver->template NextTimeStepE<TE, TB>(dt,E,B,pdE);
		};
	}

	return os.str();
}

}
// namespace simpla

#endif /* SOLVER_H_ */
