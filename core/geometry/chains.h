/*
 * chains.h
 *
 *  Created on: 2015年6月4日
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CHAINS_H_
#define CORE_GEOMETRY_CHAINS_H_
#include "primitive.h"

namespace simpla
{
namespace geometry
{
template<size_t Dimension, typename ...> struct Primitive;
template<typename ...> struct Chains;

namespace tags
{
struct is_structed;
}  // namespace tags

template<typename CoordinateSystem, typename ... Others>
using PointSet=Chains<Primitive<0, CoordinateSystem,tags::simplex>,Others...>;

/**
 * @brief Curve
 *  topological 1-dimensional geometric primitive (4.15), representing
 *   the continuous image of a line
 *  @note The boundary of a curve is the set of points at either end of the curve.
 * If the curve is a cycle, the two ends are identical, and the curve
 *  (if topologically closed) is considered to not have a boundary.
 *  The first point is called the start point, and the last is the end
 *  point. Connectivity of the curve is guaranteed by the “continuous
 *  image of a line” clause. A topological theorem states that a
 *  continuous image of a connected set is connected.
 */
template<typename ...Others, typename ...Others2>
using Curve=Chains<Primitive<1, Others...>,Others2...>;

/**
 * @brief Surface
 * topological 2-dimensional  geometric primitive (4.15),
 * locally representing a continuous image of a region of a plane
 * @note The boundary of a surface is the set of oriented, closed curves
 *  that delineate the limits of the surface.
 *
 */
template<typename ...Others, typename ...Others2>
using Surface=Chains<Primitive<2, Others...>,Others2...>;

/**
 * @brief Solids
 */
template<typename ...Others, typename ...Others2>
using Solids=Chains<Primitive<3, Others...>,Others2...>;

namespace traits
{
template<typename PrimitiveType, typename ...Others>
struct coordinate_system<Chains<PrimitiveType, Others...>>
{
	typedef typename coordinate_system<PrimitiveType>::type type;
};
template<typename PrimitiveType, typename ...Others>
struct dimension<Chains<PrimitiveType, Others...>>
{
	static constexpr size_t value = dimension<PrimitiveType>::value;
};

template<typename > struct is_structed;
template<typename PrimitiveType, typename ...Others>
struct is_structed<Chains<PrimitiveType, Others...>>
{
	static constexpr bool value =
			find_type_in_list<tags::is_structed, Others...>::value;
};
}  // namespace traits



}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_CHAINS_H_ */
