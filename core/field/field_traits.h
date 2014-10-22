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

template<typename TD, typename ...Args>
void scatter(TD const & d, Args && ... args)
{
}
template<typename TD, typename ...Args>
void gather(TD const & d, Args && ... args)
{
}

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

}  // namespace simpla

#endif /* FIELD_TRAITS_H_ */
