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
template<typename, int, typename > class Field;
template<int IFORM, typename TM, typename TV>
bool LoadField(LuaObject const &dict, Field<TM, IFORM, TV> *f)
{
	if (!dict)
		return false;

	typedef TM mesh_type;
	typedef typename Field<TM, IFORM, TV>::value_type value_type;
	typedef typename Field<TM, IFORM, TV>::field_value_type field_value_type;

	mesh_type const &mesh = f->mesh;

	if (dict.is_function())
	{
		f->Init();
		mesh.template Traversal<IFORM>(

		[&](typename mesh_type::index_type s)
		{
			auto x=mesh.GetCoordinates(s);

			auto v=dict(x[0],x[1],x[2]).template as<field_value_type>();

			(*f)[s] = mesh.Sample(Int2Type<IFORM>(),s,v);
		});

	}
	else if (dict.is_number() | dict.is_table())
	{
		f->Fill(dict.as<value_type>());
	}
	else if (dict.is_string())
	{
		std::string url = dict.as<std::string>();
		//TODO Read field from data file
		UNIMPLEMENT << "Read field from data file or other URI";

		return false;
	}

	return true;
}
}  // namespace simpla

#endif /* LOAD_FIELD_H_ */
