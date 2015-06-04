/*
 * chains.h
 *
 *  Created on: 2015年6月4日
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CHAINS_H_
#define CORE_GEOMETRY_CHAINS_H_
#include "primitive.h"
#include "../gtl/type_traits.h"
namespace simpla
{
namespace geometry
{
template<size_t Dimension, typename ...> struct Primitive;
template<typename ...> struct Chains;

namespace tags
{
struct is_structed;
}  // namespace tags

template<typename CoordinateSystem, typename ... Others>
using PointSet=Chains<Primitive<0, CoordinateSystem,tags::simplex>,Others...>;

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

namespace traits
{

template<typename > struct is_chains;
template<typename > struct is_primitive;

template<typename ...Others>
struct is_primitive<Chains<Others...>>
{
	static constexpr bool value = false;
};

template<typename ...Others>
struct is_chains<Chains<Others...>>
{
	static constexpr bool value = true;
};

template<typename PrimitiveType, typename ...Others>
struct coordinate_system<Chains<PrimitiveType, Others...>>
{
	typedef typename coordinate_system<PrimitiveType>::type type;
};
template<typename PrimitiveType, typename ...Others>
struct dimension<Chains<PrimitiveType, Others...>>
{
	static constexpr size_t value = dimension<PrimitiveType>::value;
};

template<typename > struct is_structed;
template<typename PrimitiveType, typename ...Others>
struct is_structed<Chains<PrimitiveType, Others...>>
{
	static constexpr bool value =
			find_type_in_list<tags::is_structed, Others...>::value;
};
}  // namespace traits

template<size_t Dimension, typename ...Others, typename TMesh>
struct Chains<Primitive<Dimension, Others...>, TMesh> : public std::vector<
		nTuple<typename TMesh::id_type,
				traits::template number_of_vertices<
						Primitive<Dimension, Others...> >::value>>

{
	typedef Primitive<Dimension, Others...> primitive_type;

	static constexpr size_t number_of_vertices =
			traits::template number_of_vertices<primitive_type>::value;

	typedef typename TMesh::id_type point_id_type;

	typedef typename TMesh::coordinates_type point_type;

	typedef std::vector<nTuple<point_id_type, number_of_vertices>> base_type;

	std::shared_ptr<TMesh> m_mesh_;

	primitive_type operator[](
			typename simpla::traits::key_type<base_type>::type const & n) const
	{
		primitive_type res;
		for (int i = 0; i < number_of_vertices; ++i)
		{
			res[i] = m_mesh_.coordinates(base_type::operator[](n)[i]);
		}
		return std::move(res);
	}
	using base_type::emplace_back;
	using base_type::push_back;
	using base_type::clear;
	using base_type::size;
	using base_type::begin;
	using base_type::end;
};

}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_CHAINS_H_ */
