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
#include "field_function.h"
namespace simpla
{

template<typename TDict, typename TField>
bool load_field(TDict const & dict, TField *f)
{

	if (!dict)
	{
		return false;
	}
	else if (dict.is_string())
	{
		std::string url = dict.template as<std::string>();
		//TODO Read field from data file
		UNIMPLEMENTED << "Read field from data file or other URI";

		return false;
	}

	if (!f->is_valid())
	{
		f->clear();
	}

	typedef TField field_type;

	typedef typename field_type::value_type value_type;

	typedef typename field_type::domain_type domain_type;

	domain_type domain(f->domain());

	domain.filter_by_config(dict["Domain"]);

	*f = make_field_function<value_type>(domain, dict["Value"]);

	f->sync();
	f->wait();
	return true;
}
//
//template<typename TMesh, typename TDomain, typename TDict, typename TF>
//bool assign_field_by_config_impl_(TMesh const & mesh, TDomain const & domain,
//		TDict const &dict, TF *f)
//{
//
//	typedef typename TF::field_value_type field_value_type;
//
//	if (dict.is_function())
//	{
//		for (auto const &s : domain)
//		{
//			(*f)[s] = mesh.sample(
//					dict(mesh.coordinates(s)).template as<field_value_type>(),
//					s);
//		}
//	}
//	else if (dict.is_table())
//	{
//		auto v = dict.template as<field_value_type>();
//
//		for (auto const &s : domain)
//		{
//			(*f)[s] = mesh.sample(v, s);
//		}
//
//	}
//	else
//	{
//		return false;
//	}
//
//	return true;
//
//}
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
