/**
 * context_cylindrical.cpp
 *
 * \date 2014-7-9
 * \author salmon
 */

#include <utility>

#include "../../src/mesh/geometry_cylindrical.h"
#include "../../src/mesh/mesh_rectangle.h"
#include "../../src/mesh/uniform_array.h"
#include "../../src/utilities/factory.h"
#include "../../src/utilities/lua_state.h"
#include "explicit_em.h"

namespace simpla
{
void RegisterEMContextCylindrical(Factory<std::string, ContextBase, LuaObject> * factory)
{
	factory->Register(ExplicitEMContext<Mesh<CylindricalCoordinates<SurturedMesh>>> ::template CreateFactoryFun<LuaObject>());
	factory->Register(ExplicitEMContext<Mesh<CylindricalCoordinates<SurturedMesh>,true>> ::template CreateFactoryFun<LuaObject>());
}

} // namespace simpla

