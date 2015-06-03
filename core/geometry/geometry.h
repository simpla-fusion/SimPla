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

namespace simpla
{
namespace geometry
{
typedef nTuple<Real, 2> Point2;
typedef nTuple<Real, 3> Point3;
} // namespace geometry
} // namespace simpla

#ifdef USE_BOOST
#	error "Custom geometry library is not implemented!"
#else
#	include "boost_gemetry_adapted.h"
#endif

#endif /* CORE_GEOMETRY_GEOMETRY_H_ */
