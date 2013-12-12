/*
 * load_field.h
 *
 *  Created on: 2013年12月9日
 *      Author: salmon
 */

#ifndef LOAD_FIELD_H_
#define LOAD_FIELD_H_

#include <cstddef>
#include <iostream>
#include <string>

#include "../fetl/field.h"
#include "log.h"
#include "lua_state.h"

namespace simpla
{

template<int IFORM, typename TM, typename TV>
void LoadField(LuaObject const &obj, Field<Geometry<TM, IFORM>, TV> *f)
{
	if (obj.isNull())
		return;

	typedef TM mesh_type;

	mesh_type const &mesh = f->mesh;

	if (obj.is_function())
	{
		mesh.TraversalCoordinates(IFORM,
				[&](size_t s,typename mesh_type::coordinates_type const &x)
				{
					if(IFORM==1 || IFORM==2)
					{
						(*f)[s]=obj(mesh.template GetSubComponent<IFORM>(s),x[0],x[1],x[2]).
						template as<TV>();
					}
					else
					{
						(*f)[s]=obj(x[0],x[1],x[2]).template as<TV>();
					}

				}, mesh_type::WITH_GHOSTS);
	}
	else if (obj.is_number())
	{
		*f = obj.as<TV>();
	}
	else
	{
		std::string url = obj.as<std::string>();
		//TODO Read field from data file
		UNIMPLEMENT;
	}
}
}  // namespace simpla

#endif /* LOAD_FIELD_H_ */
