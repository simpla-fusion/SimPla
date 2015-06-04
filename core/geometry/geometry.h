/**
 * @file geometry.h
 *
 * @date 2015年6月3日
 * @author salmon
 */

#ifndef CORE_GEOMETRY_GEOMETRY_H_
#define CORE_GEOMETRY_GEOMETRY_H_

#include "../gtl/primitives.h"
#include "../gtl/ntuple.h"

//namespace simpla
//{
//namespace geometry
//{
//namespace model
//{
//
//typedef nTuple<Real, 2> Point2;
//typedef nTuple<Real, 3> Point3;
//template<typename TPoint>
//Box<TPoint> make_box(TPoint const & ...p)
//{
//	return Box<TPoint>(p...);
//}
//
//template<typename TPoint>
//Polygon<TPoint> make_polygon(TPoint const & ...p)
//{
//	return Polygon<TPoint>(p...);
//}
//template<typename TPoint>
//LineSegment<TPoint> make_linesegment(TPoint const & ...p)
//{
//	return LineSegment<TPoint>(p...);
//}
//
//}  // namespace model
//
//} // namespace geometry
//} // namespace simpla

#include "coordinate_system.h"
#include "manifold.h"

#ifdef USE_BOOST
#	error "Custom geometry library is not implemented!"
#else
#	include "boost_gemetry_adapted.h"
#endif

#endif /* CORE_GEOMETRY_GEOMETRY_H_ */
