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

//template<typename CoordinateSystem>
//using Point = Primitive< 0,CoordinateSystem, tags::simplex>;

template<typename CoordinateSystem>
using LineSegment = Primitive< 1,CoordinateSystem, tags::simplex >;

template<typename CoordinateSystem>
using Triangle = Primitive< 2,CoordinateSystem, tags::simplex >;

template<typename CoordinateSystem>
using Tetrahedron = Primitive< 2,CoordinateSystem, tags::simplex >;

template<typename CoordinateSystem>
using Rectangle = Primitive< 2,CoordinateSystem, tags::cube >;

template<typename CoordinateSystem>
using Hexahedron = Primitive< 3,CoordinateSystem, tags::cube >;
/**
 * @brief Polyline a 'polygonal chain' is a connected series of 'line segments'.
 *  More formally, a 'polygonal chain' $P$ is a curve specified by a
 *  sequence of points $\scriptstyle(A_1, A_2, \dots, A_n)$ called its
 *  'vertices'. The curve itself consists of the 'line segments' connecting
 *  the consecutive vertices. A 'polygonal chain' may also be called a
 *  'polygonal curve', 'polygonal path', 'polyline',or 'piecewise linear curve'.
 *
 * Curve topological 1-dimensional geometric primitive (4.15), representing
 *   the continuous image of a line
 *  @note The boundary of a curve is the set of points at either end of the curve.
 * If the curve is a cycle, the two ends are identical, and the curve
 *  (if topologically closed) is considered to not have a boundary.
 *  The first point is called the start point, and the last is the end
 *  point. Connectivity of the curve is guaranteed by the “continuous
 *  image of a line” clause. A topological theorem states that a
 *  continuous image of a connected set is connected.
 */
template<typename CS, typename ...Others>
using Polyline=Chains<Primitive<1,CS,tags::simplex>,Others...>;

/**
 *   @brief Polygon In geometry, a polygon  is traditionally a plane
 *   figure that is bounded by a closed 'polyline'. These segments
 *    are called its edges or sides, and the points where two edges meet are the polygon's vertices (singular: vertex) or corners.
 */
template<typename CS>
struct Polygon
{
	typedef Point<CS> point_type;

	typedef Polyline<CS, tags::is_closed> ring_type;

	typedef std::vector<ring_type> inner_container_type;

	inline ring_type const& outer() const
	{
		return m_outer_;
	}

	inline ring_type& outer()
	{
		return m_outer_;
	}
	inline inner_container_type const& inners() const
	{
		return m_inners_;
	}
	inline inner_container_type & inners()
	{
		return m_inners_;
	}

	/// Utility method, clears outer and inner rings
	inline void clear()
	{
		m_outer_.clear();
		m_inners_.clear();
	}
private:
	ring_type m_outer_;
	inner_container_type m_inners_;
};
template<typename OS, typename CS>
OS & operator<<(OS & os, Polygon<CS> const & poly)
{
	os << poly.outer();
	return os;
}

/**
 * @brief Surface
 * topological 2-dimensional  geometric primitive (4.15),
 * locally representing a continuous image of a region of a plane
 * @note The boundary of a surface is the set of oriented, closed curves
 *  that delineate the limits of the surface.
 *
 */
template<typename CS, typename TAG, typename ...Others>
struct Surface
{
	typedef typename traits::point_type<CS>::type point_type;
	typedef Primitive<2, CS, TAG> primitive_type;

	static constexpr size_t max_num_of_points = traits::number_of_points<
			primitive_type>::value;
	typedef Polyline<CS, tags::is_closed> ring_type;

};

/**
 * @brief Solids
 */
template<typename CS, typename TAG, typename ...Others>
using Solids=Chains<Primitive<3, CS,TAG>, Others ...>;

}
// namespace model

namespace traits
{

template<typename CS>
struct coordinate_system<model::Polygon<CS> >
{
	typedef CS type;
};
}  // namespace traits
}  // namespace geometry

}  // namespace simpla

#endif /* CORE_GEOMETRY_MODEL_H_ */
