/*
 * pml.cpp
 *
 *  Created on: 2014年1月1日
 *      Author: salmon
 */

#include "../../../src/mesh/co_rect_mesh.h"
#include "../../../src/engine/fieldsolver.h"
#include "pml.h"
namespace simpla
{
template<typename TM>
std::shared_ptr<FieldSolver<TM> > _CreatePML(TM const & mesh)
{
	return std::dynamic_pointer_cast<FieldSolver<TM> >(std::shared_ptr<PML<TM> >(new PML<TM>(mesh)));
}
std::shared_ptr<FieldSolver<CoRectMesh<Real> > > CreatePML(CoRectMesh<Real> const & mesh)
{
	return _CreatePML(mesh);
}
std::shared_ptr<FieldSolver<CoRectMesh<Complex> > > CreatePML(CoRectMesh<Complex> const & mesh)
{

	return _CreatePML(mesh);
}

} // namespace pml
