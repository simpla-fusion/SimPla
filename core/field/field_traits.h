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

namespace simpla
{

template<typename ... >struct _Field;
template<typename ... >struct Expression;

template<typename TD, typename TOP, typename TI, typename ...Args>
auto calculate(TD const & d, TOP const & op, TI const & s, Args && ... args)
DECL_RET_TYPE((op(get_value(std::forward<Args>(args),s)...)))

template<typename TD, typename ... Others>
auto get_domain(_Field<TD, Others...> const & f)
DECL_RET_TYPE((f.domain()))

template<typename TOP, typename TL>
auto get_domain(_Field<Expression<TOP, TL> > const & expr)
DECL_RET_TYPE ((get_domain(expr.lhs) ))

template<typename TOP, typename TL, typename TR>
auto get_domain(_Field<Expression<TOP, TL, TR>> const & expr)
DECL_RET_TYPE ((get_domain(expr.lhs)&get_domain(expr.rhs)))

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


HAS_MEMBER_FUNCTION(scatter)
template<typename TD, typename TC, typename ...Args>
auto scatter(TD const & d, TC & c, Args && ... args)
ENABLE_IF_DECL_RET_TYPE( (has_member_function_scatter<TD,TC,Args...>::value),
		(d.scatter(c,std::forward<Args>(args)...)))

template<typename TD, typename TC, typename ...Args>
auto scatter(TD const & d, TC & c,
		Args && ... args)
		->typename std::enable_if<(!has_member_function_scatter<TD,TC,Args...>::value) >::type
{
}

HAS_MEMBER_FUNCTION(gather)
template<typename TD, typename TC, typename ...Args>
auto gather(TD const & d, TC & c, Args && ... args)
ENABLE_IF_DECL_RET_TYPE( (has_member_function_gather<TD,TC,Args...>::value),
		(d.gather(c,std::forward<Args>(args)...)))

template<typename TD, typename TC, typename ...Args>
auto gather(TD const & d, TC & c,
		Args && ... args)
		->typename std::enable_if<(!has_member_function_gather<TD,TC,Args...>::value) >::type
{
}

}  // namespace simpla

#endif /* FIELD_TRAITS_H_ */
