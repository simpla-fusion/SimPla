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

		auto mesh_type = dict["Grid"]["Type"].as<std::string>();
		auto scalar_type = dict["Grid"]["ScalarType"].as<std::string>("Real");

//		if (mesh_type == "CoRectMesh" && scalar_type == "Complex")
//		{
//			CreateContext<ExplicitEMContext<CoRectMesh<Complex>>>(dict,this);
//		}
//		else if (mesh_type == "CoRectMesh" && scalar_type == "Real")
//		{
//			CreateContext<ExplicitEMContext<CoRectMesh<Real>>>(dict, this);
//
//		}

		if (mesh_type == "RectMesh")
		{
			CreateContext<ExplicitEMContext<RectMesh<> >>(dict, this);

		}
		LOGGER << ">>>>>>> Initialization Load Complete! <<<<<<<< ";

	}

}

}  // namespace simpla
