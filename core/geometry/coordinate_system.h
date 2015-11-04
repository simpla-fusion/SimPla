/**
 * @file coordinate_system.h
 *
 * @date 2015-6-3
 * @author salmon
 */

#ifndef CORE_GEOMETRY_COORDINATE_SYSTEM_H_
#define CORE_GEOMETRY_COORDINATE_SYSTEM_H_

#include <stddef.h>
#include <cstdbool>
#include <type_traits>

#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"

namespace simpla
{

template<typename, typename> struct map;
template<typename...> struct Metric;


namespace coordinate_system
{

struct Polar
{
};

template<size_t N, size_t ZAXIS = 2>
struct Cartesian
{
};

struct Spherical
{
};
template<int IPhiAxis = 2>
struct Cylindrical
{
    static constexpr size_t PhiAxis = (IPhiAxis) % 3;
    static constexpr size_t RAxis = (PhiAxis + 1) % 3;
    static constexpr size_t ZAxis = (PhiAxis + 2) % 3;

};

struct Toroidal
{
};

struct MagneticFLux
{
};

} //namespace coordinate_system

namespace traits
{

template<typename> struct dimension;
template<typename> struct coordinate_type;
template<typename> struct is_homogeneous;
template<typename> struct typename_as_string;


template<typename ...> struct coordinate_system_type;
template<typename T> using coordinate_system_t= typename coordinate_system_type<T>::type;

template<typename CS, typename ...Others>
struct coordinate_system_type<Metric<CS, Others...> >
{
    typedef CS type;
};

template<typename> struct ZAxis;
template<typename T>
struct ZAxis : public ZAxis<coordinate_system_t<T>>
{
};

template<typename CS> struct scalar_type
{
    typedef Real type;
};
template<typename CS>
using scalar_type_t =typename scalar_type<CS>::type;

template<typename CS>
struct point_type
{
    typedef nTuple <Real, dimension<CS>::value> type;
};
template<typename CS>
using point_t=typename point_type<CS>::type;
template<typename CS>
using point_type_t=typename point_type<CS>::type;
template<typename CS>
struct vector_type
{
    typedef nTuple <Real, dimension<CS>::value> type;
};
template<typename CS>
using vector_t=typename vector_type<CS>::type;
template<typename CS>
using vector_type_t=typename vector_type<CS>::type;
template<typename CS>
struct covector_type
{
    typedef nTuple <Real, dimension<CS>::value> type;
};
template<typename CS>
using covector_t=typename vector_type<CS>::type;
template<typename CS>
using covector_type_t=typename vector_type<CS>::type;

template<typename CS>
struct dimension
{
    static constexpr int value = 3;
};

template<>
struct dimension<coordinate_system::Polar>
{
    static constexpr int value = 2;
};
template<int N>
struct dimension<coordinate_system::Cartesian<N>>
{
    static constexpr int value = N;
};

template<typename CS>
struct coordinate_type
{
    typedef Real type;
};
/**
 * if coordinate system is state-less value=true
 *  else value = false
 */
template<typename CS>
struct is_homogeneous
{
    static constexpr bool value = true;
};
template<>
struct is_homogeneous<coordinate_system::Toroidal>
{
    static constexpr bool value = false;
};

template<>
struct is_homogeneous<coordinate_system::MagneticFLux>
{
    static constexpr bool value = false;
};
template<int N>
struct typename_as_string<coordinate_system::Cartesian<N>>
{
    static constexpr char value[] = "Cartesian";
};
template<>
struct typename_as_string<coordinate_system::Spherical>
{
    static constexpr char value[] = "Spherical";
};

template<int PhiAXIS>
struct typename_as_string<coordinate_system::Cylindrical<PhiAXIS>>
{
    static constexpr char value[] = "Cylindrical";
};


template<int PhiAXIS>
struct ZAxis<coordinate_system::Cylindrical<PhiAXIS>> : public std::integral_constant<
        int, (PhiAXIS + 2) % 3>
{
};

template<int NDIMS, int ZAXIS>
struct ZAxis<coordinate_system::Cartesian<NDIMS, ZAXIS>> : public std::integral_constant<
        int, ZAXIS>
{
};


}
}   // namespace geometry   // namespace simpla

#endif /* CORE_GEOMETRY_COORDINATE_SYSTEM_H_ */
