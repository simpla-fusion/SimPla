/**
 * @file mesh_traits.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_GEOMETRY_TRAITS_H
#define SIMPLA_GEOMETRY_TRAITS_H

#include <type_traits>
#include <string>
#include "../../gtl/type_traits.h"
#include "../manifold_traits.h"


namespace simpla
{


template<typename ...> class Geometry;

namespace traits
{


template<typename ... T>
struct type_id<Geometry<T...> >
{
    static std::string name()
    {
        return "Geometry<" + type_id<T...>::name() + "> ";
    }
};

template<typename> struct is_geometry;

template<typename> struct geometry_type;

template<typename T> struct is_geometry : public std::integral_constant<bool, false>
{
};

template<typename ...T> struct is_geometry<Geometry<T...>> : public std::integral_constant<bool, true>
{
};

template<typename T> struct geometry_type
{
    typedef std::nullptr_t type;
};

template<typename ...T> struct id_type<Geometry<T...> >
{
    typedef std::uint64_t type;
};

template<typename CS, typename ... T>
struct coordinate_system_type<Geometry<CS, T...>>
{
    typedef CS type;
};


}  // namespace traits

} //namespace simpla

#endif //SIMPLA_GEOMETRY_TRAITS_H
