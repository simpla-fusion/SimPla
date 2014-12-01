/*
 * load_field.h
 *
 *  created on: 2013-12-9
 *      Author: salmon
 */

#ifndef LOAD_FIELD_H_
#define LOAD_FIELD_H_

#include <string>

#include "../utilities/log.h"
#include "field.h"

namespace simpla
{
template<typename ... > class _Field;

template<typename TDict, typename ...T>
bool load_field_(TDict const &dict, _Field<T...> *f)
{
	if (!dict)
		return false;

	typedef typename field_traits<_Field<T...>>::manifold_type mesh_type;
	typedef typename field_traits<_Field<T...>>::value_type value_type;
	typedef typename field_traits<_Field<T...>>::field_value_type field_value_type;
	static constexpr size_t iform = field_traits<_Field<T...>>::iform;

	auto const &domain = f->domain();

	f->clear();

	if (dict.is_function())
	{

		for (auto s : domain)
		{
			auto x = domain.coordinates(s);

			auto v = dict(x).template as<field_value_type>();

			(*f)[s] = domain.Sample(std::integral_constant<size_t, iform>(), s,
					v);
		}

	}
	else if (dict.is_number() | dict.is_table())
	{

		auto v = dict.template as<field_value_type>();

		for (auto s : domain.select(iform))
		{
			auto x = domain.get_coordinates(s);

			(*f)[s] = domain.sample(std::integral_constant<size_t, iform>(), s,
					v);
		}

	}
	else if (dict.is_string())
	{
		std::string url = dict.template as<std::string>();
		//TODO Read field from data file
		UNIMPLEMENT << "Read field from data file or other URI";

		return false;
	}

	update_ghosts(f);

	return true;
}
template<int DIMS, typename TV, typename TDict, typename ...T>
bool load_field_wrap(nTuple<std::complex<TV>, DIMS>, TDict const &dict,
		_Field<T...> *f)
{

	auto ff = make_field<nTuple<Real, DIMS>>(f->domain());

	ff.clear();

	bool success = load_field_(dict, &ff);

	if (success)
		*f = ff;

	return success;
}

template<typename TV, typename TDict, typename ... T>
bool load_field_wrap(std::complex<TV>, TDict const &dict, _Field<T...> *f)
{

	auto ff = make_field<Real>(f->domain());

	ff.clear();

	bool success = load_field_(dict, &ff);

	if (success)
		*f = ff;

	return success;
}

template<typename TV, typename TDict, typename ...T>
bool load_field_wrap(TV, TDict const &dict, _Field<T...> *f)
{
	return load_field_(dict, f);
}

template<typename TDict, typename ...T>
bool load_field(TDict const &dict, _Field<T...> *f)
{
	typedef typename field_traits<_Field<T...>>::value_type value_type;

	return load_field_wrap(value_type(), dict, f);
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

#endif /* LOAD_FIELD_H_ */
