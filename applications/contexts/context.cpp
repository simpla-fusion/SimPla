/*
 * explicit_em.cpp
 *
 * \date  2013年12月29日
 *      \author  salmon
 */

#include "context.h"

#include <complex>
#include <memory>
#include <new>
#include <string>

#include "../../src/mesh/uniform_array.h"
#include "../../src/mesh/geometry_cartesian.h"
#include "../../src/mesh/geometry_cylindrical.h"
#include "../../src/mesh/mesh_rectangle.h"
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
void CreateContext(Context* ctx, TDict const &dict, Others && ...others)
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
			CreateContext<ExplicitEMContext<Mesh<CartesianGeometry<UniformArray>, false> >>(this, dict);

		}
		else if (mesh_str == "RectMeshKz")
		{
			CreateContext<ExplicitEMContext<Mesh<CartesianGeometry<UniformArray>, true> >>(this, dict);
		}
		else if (mesh_str == "CylindricalRectMesh")
		{
			CreateContext<ExplicitEMContext<Mesh<CylindricalGeometry<UniformArray>, false> >>(this, dict);

		}
		else if (mesh_str == "CylindricalRectMeshKz")
		{
			CreateContext<ExplicitEMContext<Mesh<CylindricalGeometry<UniformArray>, true> >>(this, dict);
		}
		else
		{
			LOGGER << "Unknown Grid type: " << mesh_str;
		}

		LOGGER << ">>>>>>> Initialization Load Complete! <<<<<<<< ";

	}

}

}  // namespace simpla
