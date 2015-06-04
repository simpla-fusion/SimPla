/*
 * model.h
 *
 *  Created on: 2015年6月5日
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_MODEL_H_
#define CORE_GEOMETRY_MODEL_H_
#include <list>
#include "primitive.h"
#include "chains.h"
namespace simpla
{

namespace geometry
{

namespace model
{

template<typename CoordinateSystem>
using Point= Primitive<0, CoordinateSystem, tags::simplex>;
template<typename CoordinateSystem>
using Box = Primitive< 3,CoordinateSystem, tags::box >;
//template<typename CoordinateSystem>
//using Point = Primitive< 0,CoordinateSystem, tags::simplex>;

template<typename CoordinateSystem>
using Line = Primitive< 1,CoordinateSystem, tags::simplex >;

template<typename CoordinateSystem>
using Triangle = Primitive< 2,CoordinateSystem, tags::simplex >;

template<typename CoordinateSystem>
using Tetrahedron = Primitive< 2,CoordinateSystem, tags::simplex >;

template<typename CoordinateSystem>
using Rectangle = Primitive< 2,CoordinateSystem, tags::cube >;

template<typename CoordinateSystem>
using Cube = Primitive< 3,CoordinateSystem, tags::cube >;
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
template<typename CS, typename TAG, typename ...Others>
using Curve=Chains<Primitive<1, CS,TAG>, Others ...>;

/**
 * @brief Surface
 * topological 2-dimensional  geometric primitive (4.15),
 * locally representing a continuous image of a region of a plane
 * @note The boundary of a surface is the set of oriented, closed curves
 *  that delineate the limits of the surface.
 *
 */
template<typename CS, typename TAG, typename ...Others>
using Surface=Chains<Primitive<2, CS,TAG>, Others ...>;

/**
 * @brief Solids
 */
template<typename CS, typename TAG, typename ...Others>
using Solids=Chains<Primitive<3, CS,TAG>, Others ...>;

template<size_t N, typename CS, typename TAG, typename ...Others>
class expPolygon
{

public:

	// Member types
	typedef Primitive<0, CS, TAG> point_type;
	typedef Chains<Primitive<N, CS, TAG>, Others ...> ring_type;
	typedef std::list<ring_type> inner_container_type;

	inline ring_type const& outer() const
	{
		return m_outer;
	}
	inline inner_container_type const& inners() const
	{
		return m_inners;
	}

	inline ring_type& outer()
	{
		return m_outer;
	}
	inline inner_container_type & inners()
	{
		return m_inners;
	}

	/// Utility method, clears outer and inner rings
	inline void clear()
	{
		m_outer.clear();
		m_inners.clear();
	}

private:

	ring_type m_outer;
	inner_container_type m_inners;
};

}  // namespace model

}  // namespace geometry

}  // namespace simpla

#endif /* CORE_GEOMETRY_MODEL_H_ */
