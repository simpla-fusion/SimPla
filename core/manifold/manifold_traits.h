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
#include "../geometry/coordinate_system.h"

namespace simpla
{


enum ManifoldTypeID
{
    VERTEX = 0,

    EDGE = 1,

    FACE = 2,

    VOLUME = 3
};


template<typename ...> struct Topology;

/**
 *
 */
template<typename ...> struct Manifold;

template<typename ... T>
std::ostream &operator<<(std::ostream &os, Manifold<T...> const &d)
{
    d.print(os);

    return os;
}

template<typename ...T>
std::shared_ptr<Manifold<T...>> make_mesh()
{
    return std::make_shared<Manifold<T...>>();
}
/**
 *  Default value of Manifold are defined following
 */
namespace traits
{

template<typename> struct id_type;
template<typename T> using id_type_t= typename id_type<T>::type;


template<typename ... T>
struct type_id<Manifold<T...> >
{
    static std::string name()
    {
        return "Manifold<" + type_id<T...>::name() + " >";
    }
};

template<typename T>
struct is_manifold : public std::integral_constant<bool, false>
{
};
template<typename ...T>
struct is_manifold<Manifold<T...>> : public std::integral_constant<bool, true>
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

template<typename ...T> struct id_type<Manifold<T...> >
{
    typedef std::uint64_t type;
};


}  // namespace traits

template<typename ...> class BaseManifold;

namespace traits
{


template<typename ... T>
struct type_id<BaseManifold<T...> >
{
    static std::string name()
    {
        return std::string("BaseManifold<") + type_id<T...>::name() + std::string(">");
    }
};

template<typename> struct is_geometry;

template<typename> struct geometry_type;

template<typename T> struct is_geometry : public std::integral_constant<bool, false>
{
};

template<typename ...T> struct is_geometry<BaseManifold<T...>> : public std::integral_constant<bool, true>
{
};

template<typename T> struct geometry_type
{
    typedef std::nullptr_t type;
};

template<typename ...T> struct id_type<BaseManifold<T...> >
{
    typedef std::uint64_t type;
};


}  // namespace traits


namespace geometry
{
template<typename ...> struct Metric;

namespace traits
{
template<typename ...> struct coordinate_system_type;

template<typename TBase, typename ... T>
struct coordinate_system_type<Manifold<TBase, T...>>
{
    typedef typename coordinate_system_type<TBase>::type type;
};

template<typename CS, typename ... T0, typename ... T>
struct coordinate_system_type<BaseManifold<::simpla::geometry::Metric<CS, T0...>, T...>>
{
    typedef CS type;
};


}
} //namespace geometry // namespace traits


}  // namespace simpla

#endif /* CORE_MESH_MESH_TRAITS_H_ */
