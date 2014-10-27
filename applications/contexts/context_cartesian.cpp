/*
 * explicit_em.cpp
 *
 * \date  2013-12-29
 * \author  salmon
 */

#include <utility>
#include <string>
#include "../../core/manifold/geometry/cartesian.h"
#include "../../core/manifold/topology/structured.h"
#include "../../core/utilities/factory.h"
#include "../../core/utilities/lua_state.h"
#include "explicit_em.h"

namespace simpla
{

void RegisterEMContextCartesian(
		Factory<std::string, ContextBase, LuaObject> * factory)
{
	typedef Manifold<CartesianCoordinates<StructuredMesh>> manifold_type;
	factory->Register(
			ExplicitEMContext<manifold_type>::template CreateFactoryFun<
					LuaObject>());

	factory->Register(
			ExplicitEMContext<manifold_type>::template CreateFactoryFun<
					LuaObject>());
}

}
// namespace simpla
