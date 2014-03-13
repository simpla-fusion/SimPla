/*
 * explicit_em.cpp
 *
 *  Created on: 2013年12月29日
 *      Author: salmon
 */

#include "context.h"

#include <complex>
#include <memory>
#include <new>
#include <string>

#include "../../src/fetl/primitives.h"

//#include "../../src/mesh/co_rect_mesh_rz.h"
//#include "../../src/mesh/topology_rect.h"
#include "../../src/mesh/rect_mesh.h"
#include "../../src/mesh/octree_forest.h"
#include "../../src/mesh/geometry_euclidean.h"

#include "../../src/utilities/log.h"
#include "../../src/utilities/lua_state.h"
#include "explicit_em.h"

namespace simpla
{

Context::Context(LuaObject const & dict)
{
	Load(dict);
}
Context::Context()
{
	;
}
Context::~Context()
{
	;
}

void Context::Load(LuaObject const & dict)
{

	DumpData = [] (std::string const &)
	{	UNDEFINE_FUNCTION;};

	Save = [](std::ostream & os)->std::ostream &
	{
		return os;
	};

	NextTimeStep = []()
	{
		UNDEFINE_FUNCTION;
	};

	if (dict)
	{

		auto mesh_str = dict["Grid"]["Type"].as<std::string>();

		if (mesh_str == "RectMesh")
		{
			typedef RectMesh<OcForest, EuclideanGeometry> mesh_type;
			CreateContext<ExplicitEMContext<mesh_type>>(dict, this);

		}
		LOGGER << ">>>>>>> Initialization Load Complete! <<<<<<<< ";

	}

}

}  // namespace simpla
