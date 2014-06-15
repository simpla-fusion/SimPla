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

#include "../../src/mesh/octree_forest.h"
#include "../../src/mesh/mesh_rectangle.h"
#include "../../src/mesh/geometry_euclidean.h"

#include "../../src/utilities/primitives.h"
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
}
Context::~Context()
{
}

template<typename TC, typename TDict, typename ... Others>
void CreateContext(Context* ctx, TDict const &dict, Others const & ...others)
{

	std::shared_ptr<TC> ctx_ptr(new TC(dict, std::forward<Others>(others)...));
	using namespace std::placeholders;
	ctx->Save = std::bind(&TC::Save, ctx_ptr, _1, _2);
	ctx->NextTimeStep = std::bind(&TC::NextTimeStep, ctx_ptr);

}
void Context::Load(LuaObject const & dict)
{

	Save = [] (std::string const &,bool )->std::string
	{
		UNDEFINE_FUNCTION;
		return "";
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
			typedef Mesh<EuclideanGeometry<OcForest> > mesh_type;
			CreateContext<ExplicitEMContext<mesh_type>>(this, dict);

		}
		else
		{
			LOGGER << "Unknown Grid type: " << mesh_str;
		}

		LOGGER << ">>>>>>> Initialization Load Complete! <<<<<<<< ";

	}

}

}  // namespace simpla
