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
void LoadField(LuaObject const &obj, Field<TM, IFORM, TV> *f)
{

	f->Init();

	if (obj.empty())
	{
		f->Fill(0);
		return;
	}

	typedef TM mesh_type;
	typedef typename Field<TM, IFORM, TV>::value_type value_type;
	typedef typename Field<TM, IFORM, TV>::field_value_type field_value_type;

	mesh_type const &mesh = f->mesh;

	if (obj.is_function())
	{
		mesh.template Traversal<IFORM>(

		[&](typename mesh_type::index_type s)
		{
			auto x=mesh.GetCoordinates(s);
			auto v=obj(x[0],x[1],x[2]).template as<field_value_type>();
//			(*f)[s] = mesh.template GetWeightOnElement<IFORM>( v,s);
		});

//		if (IFORM == EDGE || IFORM == FACE)
//		{
//			mesh.SerialTraversal(IFORM,
//
//			[&](size_t s,typename mesh_type::coordinates_type const &x)
//			{
//				auto v=obj(x[0],x[1],x[2]).template as<field_value_type>();
//				(*f)[s] = mesh.template GetWeightOnElement<IFORM>( v,s);
//			});
//		}
//		else
//		{
//
//			mesh.SerialTraversal(IFORM, [&](size_t s,typename mesh_type::coordinates_type const &x)
//			{
//				(*f)[s]=obj(x[0],x[1],x[2]).template as<TV>();
//
//			});
//		}

	}
	else if (obj.is_number())
	{
		*f = obj.as<Real>();
	}
	else if (obj.is_table())
	{
//		mesh.AssignContainer(f, obj.as<field_value_type>());
	}
	else //if (obj.is_string())
	{
		std::string url = obj.as<std::string>();
		//TODO Read field from data file
		UNIMPLEMENT << "Read field from data file or other URI";
	}

}
}  // namespace simpla

#endif /* LOAD_FIELD_H_ */
