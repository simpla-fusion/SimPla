/*
 * solver.h
 *
 *  Created on: 2014年1月1日
 *      Author: salmon
 */

#ifndef SOLVER_H_
#define SOLVER_H_
#include "../../../src/engine/fieldsolver.h"
#include <memory>
#include <string>
namespace simpla
{

std::shared_ptr<FieldSolver<CoRectMesh<Real> > > CreateColdFluidEM(CoRectMesh<Real> const & mesh);
std::shared_ptr<FieldSolver<CoRectMesh<Real> > > CreatePML(CoRectMesh<Real> const & mesh);
std::shared_ptr<FieldSolver<CoRectMesh<Complex> > > CreateColdFluidEM(CoRectMesh<Complex> const & mesh);
std::shared_ptr<FieldSolver<CoRectMesh<Complex> > > CreatePML(CoRectMesh<Complex> const & mesh);

template<typename TM> std::shared_ptr<FieldSolver<TM> > CreateSolver(TM const & mesh, std::string const & name)
{
	std::shared_ptr<FieldSolver<TM> > res;
	if (name == "ColdFluid")
	{
		res = CreateColdFluidEM(mesh);
	}
	else if (name == "PML")
	{
		res = CreatePML(mesh);
	}
	return res;
}

}
// namespace simpla

#endif /* SOLVER_H_ */
