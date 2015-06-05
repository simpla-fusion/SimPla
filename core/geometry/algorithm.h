/**
 * @file algorithm.h
 *
 * @date 2015年6月5日
 * @author salmon
 */

#ifndef CORE_GEOMETRY_ALGORITHM_H_
#define CORE_GEOMETRY_ALGORITHM_H_

#include "primitive.h"
#include "chains.h"
#include "coordinate_system.h"
#include "model.h"

namespace simpla
{
namespace geometry
{
template<typename CS>
auto area(model::Polyline<CS, tags::is_closed> const & poly)
->decltype(std::declval<typename traits::coordinate_type<CS>::type>()*
		std::declval<typename traits::coordinate_type<CS>::type>())
{

}
}
// namespace geometry

}// namespace simpla

#endif /* CORE_GEOMETRY_ALGORITHM_H_ */
