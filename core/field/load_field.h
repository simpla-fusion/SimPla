/*
 * load_field.h
 *
 *  created on: 2013-12-9
 *      Author: salmon
 */

#ifndef CORE_FIELD_LOAD_FIELD_H_
#define CORE_FIELD_LOAD_FIELD_H_

#include <string>

#include "../utilities/log.h"
#include "../model/select.h"

#include "field.h"

namespace simpla
{

template<typename ... > class _Field;

template<typename TDict, typename ...T>
bool load(TDict const &dict, _Field<T...> *f)
{
	if (!f->is_valid())
	{
		f->clear();
	}
	if (!dict)
		return false;

	if (dict.is_string())
	{
		std::string url = dict.template as<std::string>();
		//TODO Read field from data file
		UNIMPLEMENTED << "Read field from data file or other URI";

		return false;
	}

	typedef _Field<T...> field_type;

	typedef typename field_type::field_value_type field_value_type;

	auto const & mesh = f->mesh();

	typedef typename field_type::mesh_type mesh_type;

	typedef typename mesh_type::id_type id_type;

	std::set<id_type> range;

//	select_by_config(mesh, dict["Select"], mesh.range(), &range);

	auto value = dict["Value"];

	if (value.is_function())
	{
		// TODO Lua.funcition object should be  parallelism

		for (auto s : range)
		{
			auto x = mesh.coordinates(s);

			auto v = value(x).template as<field_value_type>();

			(*f)[s] = mesh.sample(v, s);
		}

	}
	else if (value.is_number() | value.is_table())
	{

		auto v = value.template as<field_value_type>();

		for (auto s : range)
		{

			(*f)[s] = mesh.sample(v, s);
		}

	}

	f->sync();

	return true;
}
//template<int DIMS, typename TV, typename TDict, typename ...T>
//bool load_field_wrap(nTuple<std::complex<TV>, DIMS>, TDict const &dict,
//		_Field<T...> *f)
//{
//
//	auto ff = make_field<nTuple<Real, DIMS>>(f->domain());
//
//	ff.clear();
//
//	bool success = load_field_(dict, &ff);
//
//	if (success)
//		*f = ff;
//
//	return success;
//}
//
//template<typename TV, typename TDict, typename ... T>
//bool load_field_wrap(std::complex<TV>, TDict const &dict, _Field<T...> *f)
//{
//
//	auto ff = make_field<Real>(f->domain());
//
//	ff.clear();
//
//	bool success = load_field_(dict, &ff);
//
//	if (success)
//		*f = ff;
//
//	return success;
//}
//
//template<typename TV, typename TDict, typename ...T>
//bool load_field_wrap(TV, TDict const &dict, _Field<T...> *f)
//{
//	return load_field_(dict, f);
//}
//
//template<typename TDict, typename ...T>
//bool load_field(TDict const &dict, _Field<T...> *f)
//{
//	typedef typename field_traits<_Field<T...>>::value_type value_type;
//
//	return load_field_wrap(value_type(), dict, f);
//}

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
//	else if (dict.is_number() && !isnTuple<field_value_type>::value)
//	{
//		field_value_type v = dict.template as<field_value_type>();
//
//		Assign([v](coordinates_type )->field_value_type
//		{
//			return v;
//		});
//
//	}
//	else if (dict.is_table() && isnTuple<field_value_type>::value)
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

#endif /* CORE_FIELD_LOAD_FIELD_H_ */
