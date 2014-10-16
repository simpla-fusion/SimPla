/*
 * @file field.h
 *
 * @date  2013-7-19
 * @author  salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

#include <cstdbool>
#include <memory>

#include "base_field.h"

namespace simpla
{

/**
 *  \brief Field concept
 */
template<typename, typename > struct BaseField;
template<typename ... >struct Expression;
template<typename ... >struct _Field;

/**
 *
 *  \brief skeleton of Field data holder
 */
template<typename TDomain, typename Container,
		template<typename > class ... Policies>
struct _Field<BaseField<TDomain, Container>,
		Policies<BaseField<TDomain, Container>> ...> : public BaseField<TDomain,
		Container>, public Policies<BaseField<TDomain, Container>> ...
{

	typedef BaseField<TDomain, Container> base_type;

	typedef _Field<base_type, Policies<base_type> ...> this_type;

public:

	template<typename ...Args>
	_Field(Args &&...args) :
			base_type(std::forward<Args>(args)...), Policies<base_type>(*this)...
	{
	}
	_Field(this_type const & that) :
			base_type(that), Policies<base_type>(*this)...
	{
	}

	~_Field()
	{
	}

	using base_type::operator=;
	using base_type::operator+=;
	using base_type::operator-=;
	using base_type::operator*=;
	using base_type::operator/=;
}
;

/**
 *     \brief skeleton of Field expression
 *
 */
template<typename ... T>
struct _Field<Expression<T...>> : public Expression<T...>
{

	operator bool() const
	{
		auto d = get_domain(*this);
		return d && parallel_reduce(d, _impl::logical_and(), *this);
	}

	using Expression<T...>::Expression;

};

template<typename TDomain, typename TV, template<typename > class ... Policies> using Field=
_Field<BaseField<TDomain, std::shared_ptr<TV>> ,Policies<BaseField<TDomain, std::shared_ptr<TV>>> ...>;
}
// namespace simpla

#endif /* FIELD_H_ */
