/**
 * @file CoordinateSystem.h
 *
 * @date 2015-6-3
 * @author salmon
 */

#ifndef CORE_GEOMETRY_COORDINATE_SYSTEM_H_
#define CORE_GEOMETRY_COORDINATE_SYSTEM_H_

#include "simpla/SIMPLA_config.h"

#include <cstddef>
#include <cstdbool>
#include <type_traits>

#include "simpla/utilities/type_traits.h"

namespace simpla {
namespace geometry {
struct csCartesian;
struct csCylindrical;
struct csToroidal;
struct csSpherical;
struct csMagneticFLux;
struct csPolar;
/** @ingroup geometry
 *  @{
 */

/**
 *  @addtogroup coordinate_system Coordinates System
 *  @{
 */

/**
 *  Metric
 **/

namespace traits {

template <typename>
struct coordinate_type;

template <typename...>
struct coordinate_system_type;

template <typename T>
struct coordinate_system_type<T> {
    typedef typename T::coordinate_system_type type;
};
template <typename T>
using coordinate_system_t = typename coordinate_system_type<T>::type;

template <typename>
struct dimension;

template <typename CS>
struct dimension {
    static constexpr int value = 3;
};

template <>
struct dimension<csPolar> {
    static constexpr int value = 2;
};
template <>
struct dimension<csCartesian> {
    static constexpr int value = 3;
};

template <typename>
struct is_homogeneous;

/**
 * if topology_coordinate system is state-less value=true
 *  else value = false
 */
template <typename CS>
struct is_homogeneous {
    static constexpr bool value = true;
};
template <>
struct is_homogeneous<csToroidal> {
    static constexpr bool value = false;
};

template <>
struct is_homogeneous<csMagneticFLux> {
    static constexpr bool value = false;
};

template <typename CS>
struct scalar_type {
    typedef Real type;
};
template <typename CS>
using scalar_type_t = typename scalar_type<CS>::type;

template <typename CS>
struct point_type {
    typedef nTuple<Real, dimension<CS>::value> type;
};

template <typename CS>
using point_t = typename point_type<CS>::type;
template <typename CS>
using point_type_t = typename point_type<CS>::type;

template <typename CS>
struct vector_type {
    typedef nTuple<Real, dimension<CS>::value> type;
};
template <typename CS>
using vector_t = typename vector_type<CS>::type;
template <typename CS>
using vector_type_t = typename vector_type<CS>::type;

template <typename CS>
struct covector_type {
    typedef nTuple<Real, dimension<CS>::value> type;
};
template <typename CS>
using covector_t = typename vector_type<CS>::type;
template <typename CS>
using covector_type_t = typename vector_type<CS>::type;

}  // namespace traits

/** @}*/
}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_COORDINATE_SYSTEM_H_ */
