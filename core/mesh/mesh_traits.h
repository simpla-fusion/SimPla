/**
 * @file mesh_traits.h
 *
 * @date 2015年6月19日
 * @author salmon
 */

#ifndef CORE_MESH_MESH_TRAITS_H_
#define CORE_MESH_MESH_TRAITS_H_

#include <cstdbool>
#include <cstddef>
#include <type_traits>

#include "../geometry/coordinate_system.h"
#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"

namespace simpla
{
template<typename ...> struct Domain;
template<typename ...> struct Mesh;
namespace traits
{
template<typename T>
struct is_mesh: public std::integral_constant<bool, false>
{
};
template<typename ...T>
struct is_mesh<Mesh<T...>> : public std::integral_constant<bool, true>
{
};
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

template<typename T> struct mesh_type
{
	typedef std::nullptr_t type;
};
template<typename ...T> using mesh_t= typename mesh_type<T...>::type;

template<typename ...> struct id_type
{
	typedef size_t type;
};
template<typename ...T> using id_type_t= typename id_type<T...>::type;

template<typename T>
struct scalar_type
{
	typedef Real type;
};
template<typename ...T> using scalar_type_t= typename scalar_type<T...>::type;

template<typename > struct iform;

template<typename > struct iform: public std::integral_constant<int, 0>
{
};

template<typename ... T>
struct coordinate_system_type
{
	typedef std::nullptr_t type;
};
template<typename ...T> using coordinate_system_t= typename coordinate_system_type<T...>::type;

template<typename >
struct ZAxis: public std::integral_constant<int, 2>
{
	;
};

template<typename ...T>
struct iform_list: public integer_sequence<int, iform<T>::value...>
{
	typedef integer_sequence<int, iform<T>::value...> type;
};
template<typename ...T> using iform_list_t= typename iform_list<T...>::type;

}  // namespace traits

}  // namespace simpla

#endif /* CORE_MESH_MESH_TRAITS_H_ */
