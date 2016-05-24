/**
 * @file manifold_traits.h
 *
 * @date 2015-6-19
 * @author salmon
 */

#ifndef CORE_MESH_MESH_TRAITS_H_
#define CORE_MESH_MESH_TRAITS_H_

#include <cstdbool>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "../gtl/type_traits.h"
#include "../model/CoordinateSystem.h"

namespace simpla
{


template<typename ...> struct Topology;

/**
 *
 */
template<typename TMesh, template<typename> class ...Policies> struct Manifold;

template<typename TMesh, template<typename> class ...Policies>
std::ostream &operator<<(std::ostream &os, const Manifold<TMesh, Policies...> &m) { return m.print(os); }

template<typename TMesh, template<typename> class ...Policies>
std::shared_ptr<Manifold<TMesh, Policies...>> make_mesh()
{
    return std::make_shared<Manifold<TMesh, Policies...>>();
}
/**
 *  Default value of Manifold are defined following
 */
namespace traits
{

template<typename> struct id_type;
template<typename T> using id_type_t= typename id_type<T>::type;


template<typename TMesh, template<typename> class ...Policies>
struct type_id<Manifold<TMesh, Policies...> >
{
    static std::string name()
    {
        return "Manifold<" + type_id<TMesh>::name() + " >";
    }
};

template<typename T>
struct is_manifold : public std::integral_constant<bool, false>
{
};
template<typename TMesh, template<typename> class ...Policies>
struct is_manifold<Manifold<TMesh, Policies...>> : public std::integral_constant<bool, true>
{
};

template<typename T> struct mesh_type
{
    typedef std::nullptr_t type;
};
template<typename T> using manifold_type_t= typename mesh_type<T>::type;

template<typename T> struct id_type
{
    typedef int64_t type;
};

template<typename TMesh, template<typename> class ...Policies>
struct id_type<Manifold<TMesh, Policies...> >
{
    typedef std::uint64_t type;
};


template<typename> struct is_geometry;

template<typename> struct geometry_type;

template<typename T> struct is_geometry : public std::integral_constant<bool, false>
{
};


template<typename T> struct geometry_type
{
    typedef std::nullptr_t type;
};


template<typename ...> struct iform : public std::integral_constant<int, 0> { };

template<typename ...T>
struct iform_list : public integer_sequence<int, iform<T>::value...>
{
    typedef integer_sequence<int, iform<T>::value...> type;
};

template<typename ...T> using iform_list_t= typename iform_list<T...>::type;


}  // namespace traits


namespace geometry
{
template<typename ...> struct Metric;

namespace traits
{
template<typename ...> struct coordinate_system_type;

template<typename TM, template<typename> class ...Policies>
struct coordinate_system_type<Manifold<TM, Policies...>>
{
    typedef typename coordinate_system_type<TM>::type type;
};


}
} //namespace geometry // namespace traits


}  // namespace simpla

#endif /* CORE_MESH_MESH_TRAITS_H_ */
