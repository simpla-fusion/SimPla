/*
 * explicit_em.cpp
 *
 *  Created on: 2013年12月29日
 *      Author: salmon
 */

#include <complex>
#include <memory>
#include <new>
#include <string>

#include "../../src/engine/basecontext.h"
#include "../../src/fetl/primitives.h"
#include "../../src/mesh/co_rect_mesh.h"
#include "../../src/utilities/log.h"
#include "../../src/utilities/lua_state.h"
#include "../../src/utilities/utilities.h"
#include "explicit_em_impl.h"
namespace simpla
{

std::shared_ptr<BaseContext> CreateContextExplicitEM(LuaObject const & cfg)
{

	std::shared_ptr<BaseContext> ctx(nullptr);

	if (!cfg.empty())
	{

		LOGGER << "Initialize Context." << START;

		auto mesh_type = cfg["Grid"]["Type"].as<std::string>();
		auto scalar_type = cfg["Grid"]["ScalarType"].as<std::string>("Real");

		if (mesh_type == "CoRectMesh" && scalar_type == "Complex")
		{

			typedef CoRectMesh<Complex> mesh_type;

			std::shared_ptr<ExplicitEMContext<mesh_type>> ctx_ptr(new ExplicitEMContext<mesh_type>);

			ctx_ptr->Deserialize(cfg);

			ctx = std::dynamic_pointer_cast<BaseContext>(ctx_ptr);
		}
		else if (mesh_type == "CoRectMesh" && scalar_type == "Real")
		{
			typedef CoRectMesh<Real> mesh_type;

			std::shared_ptr<ExplicitEMContext<mesh_type>> ctx_ptr(new ExplicitEMContext<mesh_type>);

			ctx_ptr->Deserialize(cfg);

			ctx = std::dynamic_pointer_cast<BaseContext>(ctx_ptr);
		}
		LOGGER << "Initialize Context." << DONE;
	}

	return ctx;
}

}  // namespace simpla
