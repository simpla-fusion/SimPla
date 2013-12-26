/*
 * load_field.h
 *
 *  Created on: 2013年12月9日
 *      Author: salmon
 */

#ifndef LOAD_FIELD_H_
#define LOAD_FIELD_H_

#include <cstddef>
#include <string>

#include "../utilities/log.h"
#include "../utilities/lua_state.h"
#include "fetl.h"

namespace simpla
{
template<typename, typename > class Field;
template<int IFORM, typename TM, typename TV>
bool LoadField(LuaObject const &obj, Field<Geometry<TM, IFORM>, TV> *f)
{
	if (obj.empty())
		return false;

	f->Init();

	typedef TM mesh_type;
	typedef typename Field<Geometry<TM, IFORM>, TV>::value_type value_type;
	typedef typename Field<Geometry<TM, IFORM>, TV>::field_value_type field_value_type;

	mesh_type const &mesh = f->mesh;

	if (obj.is_function())
	{
		mesh.SerialTraversal(IFORM, [&](size_t s,typename mesh_type::coordinates_type const &x)
		{
			if(IFORM==1 || IFORM==2)
			{
				(*f)[s]=mesh.template GetWeightOnElement<IFORM>(
						obj(x[0],x[1],x[2]).template as<nTuple<3,TV>>(),s);
			}
			else
			{
				(*f)[s]=obj(x[0],x[1],x[2]).template as<TV>();
			}

		}, mesh_type::WITH_GHOSTS);
	}

	else if (obj.is_number())
	{
		*f = obj.as<Real>();
	}
	else if (obj.is_table())
	{
		mesh.AssignContainer(f, obj.as<field_value_type>());
	}
	else //if (obj.is_string())
	{
		std::string url = obj.as<std::string>();
		//TODO Read field from data file
		UNIMPLEMENT << "Read field from data file or other URI";
		return false;
	}

	return true;
}
}  // namespace simpla

#endif /* LOAD_FIELD_H_ */
