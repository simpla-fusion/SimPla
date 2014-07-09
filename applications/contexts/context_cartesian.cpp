/*
 * explicit_em.cpp
 *
 * \date  2013-12-29
 * \author  salmon
 */

#include <utility>
#include <string>
#include "../../src/mesh/geometry_cartesian.h"
#include "../../src/mesh/mesh_rectangle.h"
#include "../../src/mesh/uniform_array.h"
#include "../../src/utilities/factory.h"
#include "../../src/utilities/lua_state.h"
#include "explicit_em.h"

namespace simpla
{

void RegisterEMContextCartesian(Factory<std::string, ContextBase, LuaObject> * factory)
{
	factory->Register(ExplicitEMContext<Mesh<CartesianGeometry<UniformArray>>> ::template CreateFactoryFun<LuaObject>());
	factory->Register(ExplicitEMContext<Mesh<CartesianGeometry<UniformArray>,true>> ::template CreateFactoryFun<LuaObject>());
}

} // namespace simpla
