/*
 * field_traits.h
 *
 *  Created on: 2014年10月20日
 *      Author: salmon
 */

#ifndef FIELD_TRAITS_H_
#define FIELD_TRAITS_H_

#include <algorithm>
#include "../utilities/sp_type_traits.h"
#include "../utilities/constant_ops.h"
#include "../utilities/expression_template.h"

namespace simpla
{

template<typename ... >struct _Field;

template<typename TV, typename ... Others>
_Field<std::shared_ptr<TV>, Others...> make_field(Others && ...others)
{
	return std::move(
			_Field<std::shared_ptr<TV>, Others...>(
					std::forward<Others>(others)...));
}

template<typename T>
constexpr Identity get_domain(T && ...)
{
	return std::move(Identity());
}

constexpr Zero get_domain(Zero)
{
	return std::move(Zero());
}

template<typename TD> struct domain_traits
{
	typedef TD domain_type;
	typedef typename domain_type::index_type index_type;
};

HAS_MEMBER_FUNCTION(scatter)template
<typename TD, typename TC, typename ...Args>
auto scatter(TD const & d, TC & c, Args && ... args)
ENABLE_IF_DECL_RET_TYPE( (has_member_function_scatter<TD,TC,Args...>::value)
,(d.scatter(c,std::forward<Args>(args)...)))

template<typename TD, typename TC, typename ...Args>
auto scatter(TD const & d, TC & c,
		Args && ... args)
		->typename std::enable_if<(!has_member_function_scatter<TD,TC,Args...>::value) >::type
{
}

HAS_MEMBER_FUNCTION(gather)
template<typename TD, typename TC, typename ...
Args>
auto gather(TD const & d, TC & c, Args && ... args)
ENABLE_IF_DECL_RET_TYPE( (has_member_function_gather<TD,TC,Args...>::value),
		(d.gather(c,std::forward<Args>(args)...)))

template<typename TD, typename TC, typename ...Args>
auto gather(TD const & d, TC & c, Args && ... args)
->typename std::enable_if<
(!has_member_function_gather<TD,TC,Args...>::value) >::type
{
}

HAS_MEMBER_FUNCTION(calculate)

template<typename TD, typename TOP, typename ...Args>
auto calculate(TD const & d, TOP & c, Args && ... args)
ENABLE_IF_DECL_RET_TYPE(
		(has_member_function_calculate<TD,TOP,Args...>::value),
		(d.gather(c,std::forward<Args>(args)...)))

template<typename TD, typename TOP, typename ...Args>
auto calculate(TD const & d, TOP & c, Args && ... args)
ENABLE_IF_DECL_RET_TYPE(
		(!has_member_function_calculate<TD,TOP,Args...>::value),
		(op(std::forward<Args>(args)...)))

template<typename ...> struct field_traits;

template<typename TC, typename TD, typename ...Others>
struct field_traits<_Field<TC, TD, Others...>>
{
	typedef _Field<TC, TD, Others...> field_type;
	static constexpr size_t ndims = TD::ndims;

	static constexpr size_t iform = TD::iform;

	typedef TD domain_type;

	static domain_type const & get_domain(field_type const &f)
	{
		return f.domain();
	}

	static auto data(field_type & f)
	DECL_RET_TYPE((f.data()))
};

/// \defgroup   Field Expression
/// @{
template<typename ... >struct Expression;
template<typename TOP, typename TL, typename ...Others>
struct field_traits<_Field<Expression<TOP, TL, Others...> >>
{

	typedef typename field_traits<TL>::domain_type domain_type;

	static constexpr size_t ndims = domain_type::ndims;

	static constexpr size_t iform = domain_type::iform;

	static domain_type const & get_domain(
			_Field<Expression<TOP, TL, Others...>> const &f)
	{
		return field_traits<TL>::get_domain(f.lhs);
	}

	template<typename TLF, typename ...Args>
	static domain_type const & get_domain(TLF const &l, Args &&...)
	{
		return field_traits<TLF>::get_domain(l);
	}
};

/**
 *     \brief skeleton of Field expression
 */

template<typename TOP, typename TL, typename TR>
struct _Field<Expression<TOP, TL, TR>>
{
	typedef _Field<Expression<TOP, TL, TR>> this_type;
	typename _impl::reference_traits<TL>::type lhs;
	typename _impl::reference_traits<TR>::type rhs;

	typename field_traits<this_type>::domain_type domain_;

	TOP op_;

	_Field(TL const & l, TR const & r) :
			lhs(l), rhs(r), op_(), domain_(
					field_traits<this_type>::get_domain(l, r))
	{
	}
	_Field(TOP op, TL const & l, TR const & r) :
			lhs(l), rhs(r), op_(op), domain_(
					field_traits<this_type>::get_domain(l, r))
	{
	}

	~_Field()
	{
	}
	operator bool() const
	{
//		auto d = get_domain(*this);
//		return   parallel_reduce<bool>(d, _impl::logical_and(), *this);
		return false;
	}
	template<typename IndexType>
	inline auto operator[](IndexType const &s) const
	DECL_RET_TYPE ((calculate(domain_,op_, lhs, rhs, s )))

}
;

///   \brief  Unary operation
template<typename TOP, typename TL>
struct _Field<Expression<TOP, TL>>
{
	typedef _Field<Expression<TOP, TL>> this_type;
	typename _impl::reference_traits<TL>::type lhs;

	TOP op_;

	typename field_traits<this_type>::domain_type domain_;

	_Field(TL const & l) :
			lhs(l), op_(), domain_(field_traits<this_type>::get_domain(l))
	{
	}
	_Field(TOP op, TL const & l) :
			lhs(l), op_(op), domain_(field_traits<this_type>::get_domain(l))
	{
	}

	~_Field()
	{
	}

	operator bool() const
	{
		//		auto d = get_domain(*this);
		//		return   parallel_reduce<bool>(d, _impl::logical_and(), *this);
		return false;
	}

	template<typename IndexType>
	inline auto operator[](IndexType const &s) const
	DECL_RET_TYPE ((calculate(domain_,op_, lhs, s ) ))

};

DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA(_Field)
/// @}

}// namespace simpla

#endif /* FIELD_TRAITS_H_ */
