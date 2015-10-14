/**
 * @file algorithm.h
 *
 * @date 2015-6-5
 * @author salmon
 */

#ifndef CORE_GEOMETRY_ALGORITHM_H_
#define CORE_GEOMETRY_ALGORITHM_H_

#include "primitive.h"
#include "chains.h"
#include "coordinate_system.h"
#include "model.h"

namespace simpla {

namespace geometry {
template<typename CS>
auto area(model::Polyline<CS, tags::is_closed> const &poly)
-> decltype(std::declval<typename traits::coordinate_type<CS>::type>() *
            std::declval<typename traits::coordinate_type<CS>::type>())
{

}

template<typename CS, typename TGeoObject>
model::Polyline<CS> reflect(model::LineSegment<CS> const &poly, TGeoObject const &obj)
{

}

}// namespace geometry

}// namespace simpla

#endif /* CORE_GEOMETRY_ALGORITHM_H_ */
