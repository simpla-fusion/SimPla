/**
 * @file simplex.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_SIMPLEX_H_
#define CORE_GEOMETRY_SIMPLEX_H_
#include "primitive.h"

namespace simpla
{
namespace model
{
namespace model
{
template<int, typename ...T> struct Primitive;
} // namespace model

namespace tags
{
struct simplex;
} // namespace tags

namespace traits
{

template<typename > struct facet;
template<typename > struct number_of_points;

template<typename CS>
struct facet<model::Primitive<1, CS, tags::simplex>>
{
	typedef model::Primitive<0, CS> type;
};

template<typename CS>
struct facet<model::Primitive<2, CS, tags::simplex>>
{
	typedef model::Primitive<1, CS> type;
};

template<size_t N, typename CoordinateSystem>
struct number_of_points<model::Primitive<N, CoordinateSystem, tags::simplex> >
{
	static constexpr size_t value = N + 1;
};

} // namespace traits
template<typename CS>
typename traits::length_type<CS>::type distance(
		model::Primitive<0, CS> const & p,
		model::Primitive<1, CS> const & line_segment)
{

}
template<typename CS>
typename traits::length_type<CS>::type distance(
		model::Primitive<0, CS> const & p,
		model::Primitive<2, CS, tags::simplex> const & tri)
{

}
template<typename CS>
typename traits::length_type<CS>::type length(
		model::Primitive<2, CS, tags::simplex> const & tri)
{
}
template<typename CS>
typename traits::area_type<CS>::type area(
		model::Primitive<2, CS, tags::simplex> const & tri)
{
}
template<typename CS>
typename traits::volume_type<CS>::type volume(
		model::Primitive<3, CS, tags::simplex> const & poly)
{
}
}  // namespace model
}  // namespace simpla

#endif /* CORE_GEOMETRY_SIMPLEX_H_ */
