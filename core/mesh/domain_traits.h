/**
 * @file domain_traits.h
 *
 * @date 2015-6-23
 * @author salmon
 */

#ifndef CORE_MESH_DOMAIN_TRAITS_H_
#define CORE_MESH_DOMAIN_TRAITS_H_

#include <type_traits>

namespace simpla
{
template<typename ...> struct Domain;

template<int IFORM, typename ...Policies, typename TM>
Domain<TM, std::integral_constant<int, IFORM>, Policies...> make_domain(
		TM const & mesh)
{
	return Domain<TM, std::integral_constant<int, IFORM>, Policies...>(mesh);
}
namespace traits
{

template<typename > struct is_domain;
template<typename > struct domain_type;
template<typename > struct mesh_type;
template<typename > struct iform;
template<typename > struct rank;

template<typename T>
struct is_domain: public std::integral_constant<bool, false>
{
};
template<typename ...T>
struct is_domain<Domain<T...>> : public std::integral_constant<bool, true>
{
};

template<typename T> struct domain_type
{
	typedef std::nullptr_t type;
};
template<typename ...T> using domain_t= typename domain_type<T...>::type;

template<typename TM, typename ...Others>
struct mesh_type<Domain<TM, Others...> >
{
	typedef TM type;
};

template<typename > struct iform: public std::integral_constant<int, 0>
{
};

template<typename ...T>
struct iform_list: public integer_sequence<int, iform<T>::value...>
{
	typedef integer_sequence<int, iform<T>::value...> type;
};

template<typename ...T> using iform_list_t= typename iform_list<T...>::type;

template<typename TM, int IFORM, typename ...Others>
struct iform<Domain<TM, std::integral_constant<int, IFORM>, Others...> > : public std::integral_constant<
		int, IFORM>
{
};

template<typename ...T>
struct rank<Domain<T...> > : public rank<typename mesh_type<Domain<T...> >::type>::type
{
};

} // namespace traits

}  // namespace simpla

#endif /* CORE_MESH_DOMAIN_TRAITS_H_ */
