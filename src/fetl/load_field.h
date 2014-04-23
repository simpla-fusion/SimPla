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
#include "fetl.h"

namespace simpla
{
template<typename, int, typename > class Field;
template<typename TDict, int IFORM, typename TM, typename TV>
bool LoadField(TDict const &dict, Field<TM, IFORM, TV> *f)
{
	if (!dict)
		return false;

	typedef TM mesh_type;
	typedef typename Field<TM, IFORM, TV>::value_type value_type;
	typedef typename Field<TM, IFORM, TV>::field_value_type field_value_type;

	mesh_type const &mesh = f->mesh;

	f->Clear();

	if (dict.is_function())
	{

		for (auto s : mesh.GetRange(IFORM))
		{
			auto x = mesh.GetCoordinates(s);

			auto v = dict(x).template as<field_value_type>();

			(*f)[s] = mesh.Sample(Int2Type<IFORM>(), s, v);
		}

	}
	else if (dict.is_number() | dict.is_table())
	{

		auto v = dict.template as<field_value_type>();

		for (auto s : mesh.GetRange(IFORM))
		{
			auto x = mesh.GetCoordinates(s);

			(*f)[s] = mesh.Sample(Int2Type<IFORM>(), s, v);
		}

	}
	else if (dict.is_string())
	{
		std::string url = dict.template as<std::string>();
		//TODO Read field from data file
		UNIMPLEMENT << "Read field from data file or other URI";

		return false;
	}

	return true;
}

//template<typename TDict>
//void AssignFromDict(TDict const & dict)
//{
//	Clear();
//
//	if (dict.is_function())
//	{
//
//		Assign(
//
//		[dict](coordinates_type x)->field_value_type
//		{
//			return dict(x[0],x[1],x[2]).template as<field_value_type>();
//		}
//
//		);
//
//	}
//	else if (dict.is_number() && !is_nTuple<field_value_type>::value)
//	{
//		field_value_type v = dict.template as<field_value_type>();
//
//		Assign([v](coordinates_type )->field_value_type
//		{
//			return v;
//		});
//
//	}
//	else if (dict.is_table() && is_nTuple<field_value_type>::value)
//	{
//		field_value_type v = dict.template as<field_value_type>();
//
//		Assign([v](coordinates_type )->field_value_type
//		{
//			return v;
//		});
//
//	}
//	else
//	{
//		WARNING << "Can not assign field from 'dict'!";
//	}
//}
}// namespace simpla

#endif /* LOAD_FIELD_H_ */
