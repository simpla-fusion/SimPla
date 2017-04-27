/**
 * @file geometry.h
 *
 * @date 2015-6-3
 * @author salmon
 */

#ifndef CORE_GEOMETRY_GEOMETRY_H_
#define CORE_GEOMETRY_GEOMETRY_H_

#include "simpla/sp_def.h"
#include "simpla/utilities/nTuple.h"

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

template<typename TL, typename TBox>
bool in_set(TL const &l, TBox const &b) { return in_box(l, ::simpla::traits::get<0>(b), ::simpla::traits::get<1>(b)); }
}



/*  @} */
}}//namespace simpla { namespace geometry


#endif /* CORE_GEOMETRY_GEOMETRY_H_ */
