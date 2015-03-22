/*
 * explicit_em.cpp
 *
 * \date  2013-12-29
 * \author  salmon
 */

#include <utility>
#include <string>

#include "../../core/design_pattern/factory.h"
#include "../../core/diff_geometry/diff_scheme/fdm.h"
#include "../../core/diff_geometry/geometry/cartesian.h"
#include "../../core/diff_geometry/interpolator/interpolator.h"
#include "../../core/diff_geometry/mesh.h"
#include "../../core/diff_geometry/topology/structured.h"
#include "../../core/utilities/lua_state.h"
#include "explicit_em.h"

namespace simpla
{

void RegisterEMContextCartesian(
		Factory<std::string, ContextBase, LuaObject> * factory)
{
	typedef Manifold<CartesianCoordinates<RectMesh>> manifold_type;

	factory->Register(
			ExplicitEMContext<manifold_type>::template CreateFactoryFun<
					LuaObject>());

	factory->Register(
			ExplicitEMContext<manifold_type>::template CreateFactoryFun<
					LuaObject>());
}

}
// namespace simpla
