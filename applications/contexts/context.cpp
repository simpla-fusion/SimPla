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

#include "../../src/mesh/octree_forest.h"
#include "../../src/mesh/mesh_rectangle.h"
#include "../../src/mesh/geometry_cylindrical.h"
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

	Dump = [] (std::string const &)
	{
		UNDEFINE_FUNCTION;
	};

	Print = [](std::ostream & os)->std::ostream &
	{
		UNDEFINE_FUNCTION;
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
		else
		{
			LOGGER << "Unknown Grid type: " << mesh_str;
		}

		LOGGER << ">>>>>>> Initialization Load Complete! <<<<<<<< ";

	}

}

}  // namespace simpla
