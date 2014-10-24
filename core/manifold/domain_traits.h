/*
 * domain_traits.h
 *
 *  Created on: Oct 16, 2014
 *      Author: salmon
 */

#ifndef DOMAIN_TRAITS_H_
#define DOMAIN_TRAITS_H_

#include <algorithm>
#include "../utilities/sp_type_traits.h"
#include "../utilities/constant_ops.h"

namespace simpla
{
namespace _impl
{

struct Gather
{

};
struct Scatter
{

};
}  // namespace _impl

template<typename T>
constexpr Identity get_domain(T && ...)
{
	return std::move(Identity());
}

constexpr Zero get_domain(Zero)
{
	return std::move(Zero());
}

constexpr Identity operator &(Identity, Identity)
{
	return Identity();
}
template<typename TL>
TL const & operator &(TL const & l, Identity)
{
	return l;
}
template<typename TR>
TR const & operator &(Identity, TR const & r)
{
	return r;
}

template<typename TL>
constexpr Zero operator
&(TL const & l, Zero)
{
	return std::move
(Zero());
}
template<typename TR>
constexpr Zero operator &(Zero, TR const & l)
{
	return std::move(Zero());
}

template<typename TR>
constexpr Zero operator &(Zero, Zero)
{
	return std::move(Zero());
}

//template<typename TD, typename TOP, typename TI, typename ...Args>
//auto calculate(TD const & d, TOP const & op, TI const & s, Args && ... args)
//DECL_RET_TYPE((op(get_value(std::forward<Args>(args),s)...)))
//
//template<typename TD, typename ...Args>
//void calculate(TD const & d, _impl::Scatter const & op, Args && ... args)
//{
//}
//template<typename TD, typename ...Args>
//void calculate(TD const & d, _impl::Gather const & op, Args && ... args)
//{
//}
}// namespace simpla

#endif /* DOMAIN_TRAITS_H_ */
