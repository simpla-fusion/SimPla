/*
 * load_field.h
 *
 *  Created on: 2013年12月9日
 *      Author: salmon
 */

#ifndef LOAD_FIELD_H_
#define LOAD_FIELD_H_

#include "lua_state.h"
#include "log.h"
#include "../fetl/fetl.h"

namespace simpla
{

template<int IFORM, typename TM, typename TV>
void LoadField(LuaObject &obj, Field<Geometry<TM, IFORM>, TV> *f)
{

	typedef TM mesh_type;

	mesh_type const &mesh = f->mesh;

	if (obj.is_function())
	{
		mesh.TraversalCoordinates(IFORM,
				[&](size_t s,typename mesh_type::coordinates_type const &x)
				{
					if(IFORM==1 || IFORM==2)
					{
						(*f)[s]=n_obj(mesh.GetSubComponent<IFORM>(s),x[0],x[1],x[2]).as<TV>();
					}
					else
					{
						(*f)[s]=n_obj(x[0],x[1],x[2]).as<TV>();
					}

				}, mesh_type::WITH_GHOSTS);
	}
	else if (obj.is_string())
	{
		std::string url = obj.as<std::string>();
		//TODO Read field from data file
		WARNING << "UNIMPLEMENT :Read field from data file."
	}
	else
	{
		*f = obj.as<TV>();
	}
}
}  // namespace simpla

#endif /* LOAD_FIELD_H_ */
