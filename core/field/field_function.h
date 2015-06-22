/*
 * @file field_function.h
 *
 *  Created on: 2015年3月10日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_FUNCTION_H_
#define CORE_FIELD_FIELD_FUNCTION_H_

#include <stddef.h>
#include <cstdbool>
#include <functional>

#include "../gtl/primitives.h"
#include "field_traits.h"

namespace simpla
{
template<typename ...>class _Field;
template<typename ...>class Domain;

template<typename ...TDomain, typename TV, typename TFun>
class _Field<Domain<TDomain...>, TV, tags::function, TFun>
{
public:
	typedef Domain<TDomain...> domain_type;

	typedef TV value_type;
	typedef TFun function_type;

	typedef _Field<domain_type, value_type, tags::function, function_type> this_type;

	typedef typename domain_type::mesh_type mesh_type;

	typedef typename mesh_type::id_type id_type;

	typedef typename mesh_type::point_type point_type;

	typedef traits::field_value_t<this_type> field_value_type;

	static constexpr size_t iform = domain_type::iform;
	static constexpr size_t ndims = domain_type::ndims;

private:
	function_type m_fun_;
	domain_type m_domain_;
public:
	_Field()
	{
	}
	_Field(domain_type const& domain) :
			m_domain_(domain)
	{
	}
	template<typename TF>
	_Field(domain_type const& domain, TF const& fun) :
			m_domain_(domain), m_fun_(fun)
	{
	}
	_Field(this_type const& other) :
			m_domain_(other.m_domain_), m_fun_(other.m_fun_)
	{
	}
	_Field(this_type && other) :
			m_domain_(other.m_domain_), m_fun_(other.m_fun_)
	{
	}
	~_Field()
	{
	}

	bool is_valid() const
	{
		return (!!m_fun_) && m_domain_.is_valid();
	}
	operator bool() const
	{
		return !!m_fun_;
	}
	domain_type const & domain() const
	{
		return m_domain_;
	}

	value_type operator[](id_type s) const
	{
		Real t = domain().time();

		return domain().sample(s,
				static_cast<field_value_type>(m_fun_(domain().point(s), t)));
	}

	field_value_type operator()(point_type const& x, Real t) const
	{
		return static_cast<field_value_type>(m_fun_(x, t));
	}

	template<typename ...Others>
	field_value_type operator()(point_type const& x, Real t,
			Others &&... others) const
	{
		return static_cast<field_value_type>(m_fun_(x, t,
				std::forward<Others>(others )...));
	}

	/**
	 *
	 * @param args
	 * @return (x,t) -> m_fun_(x,t,args(x,t))
	 */
	template<typename ...Args>
	_Field<domain_type, value_type, tags::function,
			std::function<field_value_type(point_type const&, Real)>> op_on(
			Args const& ...args) const
	{
		typedef std::function<field_value_type(point_type const&, Real)> res_function_type;
		typedef _Field<domain_type, value_type, tags::function,
				res_function_type> res_type;

		res_function_type fun = [ &](point_type const& x, Real t)
		{
			return static_cast<field_value_type>(m_fun_(x, t,
							static_cast<field_value_type>( (args)( x ))...));
		};

		return res_type(m_domain_, fun);

	}

};

template<typename TV, typename TDomain, typename TFun>
_Field<TDomain, TV, tags::function, TFun> make_field_function(
		TDomain const& domain, TFun const& fun)
{
	return std::move(_Field<TDomain, TV, tags::function, TFun>(domain, fun));
}

template<typename TV, typename TD, typename TDict>
_Field<TD, TV, tags::function, TDict> //
make_function_by_config(TDict const & dict, TD domain)
{
	typedef TV value_type;

	typedef TD domain_type;

	typedef _Field<domain_type, value_type, tags::function, TDict> field_type;

	// TODO create null filed

	if (dict["Domain"])
	{

		filter_by_config(dict["Domain"], &domain);

		return field_type(domain, dict["Value"]);
	}
	else
	{
		domain.clear();
		return field_type(domain);

	}

}

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_FUNCTION_H_ */
