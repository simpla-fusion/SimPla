/**
 * context_cylindrical.cpp
 *
 * \date 2014-7-9
 * \author salmon
 */

#include <utility>

#include "../../core/design_pattern/factory.h"
#include "../../core/diff_geometry/domain.h"
#include "../../core/diff_geometry/geometry/cylindrical.h"
#include "../../core/diff_geometry/mesh.h"
#include "../../core/diff_geometry/topology/structured.h"
#include "../../core/manifold/geometry/cylindrical.h"
#include "../../core/utilities/lua_state.h"
#include "explicit_em.h"

namespace simpla
{
void RegisterEMContextCylindrical(
		Factory<std::string, ContextBase, LuaObject> * factory)
{
	typedef Manifold<CylindricalCoordinate<RectMesh>> manifold_type;

	factory->Register(
			ExplicitEMContext<manifold_type>::template CreateFactoryFun<
					LuaObject>());
}

} // namespace simpla

