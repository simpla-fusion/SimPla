/**
 * @file CoordinateSystem.h
 *
 * @date 2015-6-3
 * @author salmon
 */

#ifndef CORE_GEOMETRY_COORDINATE_SYSTEM_H_
#define CORE_GEOMETRY_COORDINATE_SYSTEM_H_

#include <stddef.h>
#include <cstdbool>
#include <type_traits>

#include "../sp_def.h"
#include "../gtl/type_traits.h"

namespace simpla { namespace geometry
{

/** @ingroup geometry
 *  @{
 */


/**
 *  @addtogroup coordinate_system Coordinates System
 *  @{
 */

/**
 *  coordinate system
 */
namespace coordinate_system
{

struct Polar
{
};

template<int N, int ZAXIS = 2>
struct Cartesian
{
};

struct Spherical
{
};
template<int IPhiAxis = 2>
struct Cylindrical
{
    static constexpr int PhiAxis = (IPhiAxis) % 3;
    static constexpr int RAxis = (PhiAxis + 1) % 3;
    static constexpr int ZAxis = (PhiAxis + 2) % 3;

};

struct Toroidal
{
};

struct MagneticFLux
{
};

} //namespace coordinate_system

/** @}*/

/**
 *  Metric
 **/
template<typename...> struct Metric;


namespace traits
{


template<typename> struct coordinate_type;

template<typename ...> struct coordinate_system_type;

template<typename T> struct coordinate_system_type<T>
{
    typedef typename T::coordinate_system_type type;
};
template<typename T> using coordinate_system_t= typename coordinate_system_type<T>::type;




template<typename> struct dimension;


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


template<typename> struct is_homogeneous;

/**
 * if coordinate system is state-less value=true
 *  else value = false
 */
template<typename CS> struct is_homogeneous
{
    static constexpr bool value = true;
};
template<> struct is_homogeneous<coordinate_system::Toroidal>
{
    static constexpr bool value = false;
};

template<>
struct is_homogeneous<coordinate_system::MagneticFLux>
{
    static constexpr bool value = false;
};


template<typename CS> struct scalar_type
{
    typedef Real type;
};
template<typename CS> using scalar_type_t =typename scalar_type<CS>::type;

template<typename CS> struct point_type
{
    typedef nTuple <Real, dimension<CS>::value> type;
};

template<typename CS> using point_t=typename point_type<CS>::type;
template<typename CS> using point_type_t=typename point_type<CS>::type;


template<typename CS>
struct vector_type
{
    typedef nTuple <Real, dimension<CS>::value> type;
};
template<typename CS> using vector_t=typename vector_type<CS>::type;
template<typename CS> using vector_type_t=typename vector_type<CS>::type;

template<typename CS>
struct covector_type
{
    typedef nTuple <Real, dimension<CS>::value> type;
};
template<typename CS> using covector_t=typename vector_type<CS>::type;
template<typename CS> using covector_type_t=typename vector_type<CS>::type;


template<typename CS, typename ...Others>
struct coordinate_system_type<Metric<CS, Others...> >
{
    typedef CS type;
};

template<typename ...> struct metric_type;
template<typename ...T> using metric_t=typename metric_type<T...>::type;

}// namespace traits

/** @}*/
}}  //namespace geometry   // namespace simpla

#endif /* CORE_GEOMETRY_COORDINATE_SYSTEM_H_ */
