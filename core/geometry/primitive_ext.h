/**
 * @file primitive_ext.h
 *
 *  Created on: 2015年6月4日
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_PRIMITIVE_EXT_H_
#define CORE_GEOMETRY_PRIMITIVE_EXT_H_
#include "primitive.h"
namespace simpla
{
namespace geometry
{

namespace tags
{
// linear
struct wedge;
struct pyramid;

struct infinite_wedge;
struct infinite_cube;

// quadratic
struct spline;
struct arc;

}  // namespace tags

template<size_t Dimension, typename ...> struct Primitive;

template<typename CoordinateSystem, typename Tag>
using Pyramid = Primitive< 3,CoordinateSystem, tags::pyramid >;

template<typename CoordinateSystem, typename Tag>
using Wedge = Primitive< 3,CoordinateSystem, tags::wedge>;

namespace traits
{

template<size_t Dimension, typename CoordinateSystem>
struct number_of_vertices<Primitive<Dimension, CoordinateSystem, tags::pyramid>>
{
	static constexpr size_t value = 5;
};
template<size_t Dimension, typename CoordinateSystem>
struct number_of_vertices<Primitive<Dimension, CoordinateSystem, tags::wedge>>
{
	static constexpr size_t value = 6;
};

} // namespace traits
}  //namespace geometry
}  //namespace simpla

#endif /* CORE_GEOMETRY_PRIMITIVE_EXT_H_ */
