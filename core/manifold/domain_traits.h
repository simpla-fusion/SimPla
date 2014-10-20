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

HAS_MEMBER_FUNCTION(gather);

HAS_MEMBER_FUNCTION(scatter);

template<typename domain_type, typename ...Args>
static auto gather(domain_type const &domain,
		Args && ... args)
				ENABLE_IF_DECL_RET_TYPE(( has_member_function_gather<domain_type,Args...>::value),
						(domain.gather(std::forward<Args>(args)...)))

template<typename domain_type, typename ...Args>
static auto scatter(domain_type const &domain,
		Args && ... args)
				ENABLE_IF_DECL_RET_TYPE(( has_member_function_scatter<domain_type,Args...>::value),
						(domain.scatter(std::forward<Args>(args)...)))

template<typename domain_type, typename ...Args>
static auto gather(domain_type const & domain,
		Args && ...args)
				ENABLE_IF_DECL_RET_TYPE((!has_member_function_gather<domain_type, Args...>::value),(0))

template<typename domain_type, typename ...Args>
static auto scatter(domain_type const & domain,
		Args && ...args)
				ENABLE_IF_DECL_RET_TYPE((!has_member_function_scatter<domain_type, Args...>::value),(0))

template<typename ... T>
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
constexpr Zero operator &(TL const & l, Zero)
{
	return std::move(Zero());
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
}  // namespace simpla

#endif /* DOMAIN_TRAITS_H_ */
