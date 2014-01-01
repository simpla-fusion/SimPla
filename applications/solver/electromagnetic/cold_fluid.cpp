/*
 * cold_fluid.cpp
 *
 *  Created on: 2014年1月1日
 *      Author: salmon
 */

#include "../../../src/mesh/co_rect_mesh.h"
#include "../../../src/engine/fieldsolver.h"
#include "cold_fluid.h"
namespace simpla
{

template<typename TM>
std::shared_ptr<FieldSolver<TM> > _CreateColdFluidEM(TM const & mesh)
{
	return std::dynamic_pointer_cast<FieldSolver<TM> >(std::shared_ptr<ColdFluidEM<TM> >(new ColdFluidEM<TM>(mesh)));

}
std::shared_ptr<FieldSolver<CoRectMesh<Real> > > CreateColdFluidEM(CoRectMesh<Real> const & mesh)
{
	return _CreateColdFluidEM(mesh);
}
std::shared_ptr<FieldSolver<CoRectMesh<Complex> > > CreateColdFluidEM(CoRectMesh<Complex> const & mesh)
{

	return _CreateColdFluidEM(mesh);
}

}
// namespace simpla

