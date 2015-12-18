/**
 * @file geometry.h
 *
 * @date 2015-6-3
 * @author salmon
 */

#ifndef CORE_GEOMETRY_GEOMETRY_H_
#define CORE_GEOMETRY_GEOMETRY_H_

#include "../gtl/primitives.h"
#include "../gtl/ntuple.h"

#include "CoordinateSystem.h"


#include "GeoObject.h"
#include "GeoAlgorithm.h"

namespace simpla { namespace geometry
{
/**

 *  @defgroup geometry Geometry
 *  @brief this module collects computational geometry stuff.
  */
class GeoObject;

namespace traits
{

template<typename TL> bool in_set(TL const &l, GeoObject const &r) { return r.within(l); }

template<typename TL, typename TP>
bool in_set(TL const &l, std::tuple<TP, TP> const &b) { return in_box(l, std::get<0>(b), std::get<1>(b)); }
}



/*  @} */
}}//namespace simpla { namespace geometry


#endif /* CORE_GEOMETRY_GEOMETRY_H_ */
