/**
 * @file loadField.h
 *
 *  created on: 2013-12-9
 *      Author: salmon
 */

#ifndef COREField_LOADField_H_
#define COREField_LOADField_H_

#include <string>

#include "../gtl/utilities/log.h"
#include "../geometry/select.h"
#include "field_function.h"
namespace simpla
{

template<typename TDict, typename TField>
bool loadField(TDict const & dict, TField *f)
{
	if (!f->is_valid())
	{
		f->clear();
	}

	if (dict.is_string())
	{
		std::string url = dict.template as<std::string>();
		//TODO Read field from data file
		UNIMPLEMENTED<< "Read field from data file or other URI";
	}
	else if (dict)
	{

		typedef TField field_type;

		typedef typename field_type::value_type value_type;

		typedef typename field_type::domain_type domain_type;

		domain_type domain(f->domain());

		filter_by_config(dict["Domain"], &domain);

		*f = make_function_by_config<value_type>( dict["Value"],domain);

	}

	f->sync();
	f->wait();
	return true;
}
//
//template<typename TMesh, typename TDomain, typename TDict, typename TF>
//bool assignField_by_config_impl_(TMesh const & geometry, TDomain const & domain,
//		TDict const &dict, TF *f)
//{
//
//	typedef typename TF::field_value_type field_value_type;
//
//	if (dict.is_function())
//	{
//		for (auto const &s : domain)
//		{
//			(*f)[s] = geometry.generator(
//					dict(geometry.coordinates(s)).template as<field_value_type>(),
//					s);
//		}
//	}
//	else if (dict.is_table())
//	{
//		auto v = dict.template as<field_value_type>();
//
//		for (auto const &s : domain)
//		{
//			(*f)[s] = geometry.generator(v, s);
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
//bool loadField_wrap(nTuple<std::complex<TV>, DIMS>, TDict const &dict,
//		Field<T...> *f)
//{
//
//	auto ff = make_field<nTuple<Real, DIMS>>(f->domain());
//
//	ff.clear();
//
//	bool success = loadField_(dict, &ff);
//
//	if (success)
//		*f = ff;
//
//	return success;
//}
//
//template<typename TV, typename TDict, typename ... T>
//bool loadField_wrap(std::complex<TV>, TDict const &dict, Field<T...> *f)
//{
//
//	auto ff = make_field<Real>(f->domain());
//
//	ff.clear();
//
//	bool success = loadField_(dict, &ff);
//
//	if (success)
//		*f = ff;
//
//	return success;
//}
//
//template<typename TV, typename TDict, typename ...T>
//bool loadField_wrap(TV, TDict const &dict, Field<T...> *f)
//{
//	return loadField_(dict, f);
//}
//
//template<typename TDict, typename ...T>
//bool loadField(TDict const &dict, Field<T...> *f)
//{
//	typedef typename field_traits<Field<T...>>::value_type value_type;
//
//	return loadField_wrap(value_type(), dict, f);
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
//		[dict](coordinate_tuple x)->field_value_type
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
//		Assign([v](coordinate_tuple )->field_value_type
//		{
//			return v;
//		});
//
//	}
//	else if (dict.is_table() && isnTuple<field_value_type>::value)
//	{
//		field_value_type v = dict.template as<field_value_type>();
//
//		Assign([v](coordinate_tuple )->field_value_type
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

#endif /* COREField_LOADField_H_ */
