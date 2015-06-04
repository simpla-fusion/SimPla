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

namespace tags
{
struct is_structed;
struct is_closured;
struct is_clockwise;
struct is_unordered;
}  // namespace tags

namespace model
{

template<typename ...> struct Chains;

template<typename TPrimitive, typename TMesh, typename TContainer,
		typename ...Polycies>
struct Chains<TPrimitive, TMesh, TContainer, Polycies...> : public TContainer
//public std::vector<
//		nTuple<typename TMesh::id_type,
//				traits::template number_of_vertices<
//						Primitive<Dimension, Others...> >::value>>

{
	typedef TPrimitive primitive_type;
	typedef TContainer container_type;

	static constexpr size_t number_of_vertices =
			traits::template number_of_vertices<primitive_type>::value;

	typedef typename TMesh::id_type point_id_type;

	typedef typename TMesh::coordinates_type point_type;

	std::shared_ptr<TMesh> m_mesh_;

	primitive_type operator[](
			typename simpla::traits::key_type<container_type>::type const & n) const
	{
		primitive_type res;
		for (int i = 0; i < number_of_vertices; ++i)
		{
			res[i] = m_mesh_.coordinates(container_type::operator[](n)[i]);
		}
		return std::move(res);
	}
	using container_type::emplace_back;
	using container_type::push_back;
	using container_type::clear;
	using container_type::size;
	using container_type::begin;
	using container_type::end;
};



}  // namespace model
namespace traits
{

template<typename > struct is_chains;
template<typename > struct is_primitive;
template<typename > struct is_structed;
template<typename > struct closure;
template<typename > struct point_order;

template<typename ...Others>
struct is_primitive<model::Chains<Others...>>
{
	static constexpr bool value = false;
};

template<typename ...Others>
struct is_chains<model::Chains<Others...>>
{
	static constexpr bool value = true;
};

template<typename PrimitiveType, typename ...Others>
struct coordinate_system<model::Chains<PrimitiveType, Others...>>
{
	typedef typename coordinate_system<PrimitiveType>::type type;
};
template<typename PrimitiveType, typename ...Others>
struct dimension<model::Chains<PrimitiveType, Others...>>
{
	static constexpr size_t value = dimension<PrimitiveType>::value;
};

template<typename PrimitiveType, typename ...Others>
struct is_structed<model::Chains<PrimitiveType, Others...>>
{
	static constexpr bool value =
			find_type_in_list<tags::is_structed, Others...>::value;
};
template<typename PrimitiveType, typename ...Others>
struct point_order<model::Chains<PrimitiveType, Others...>>
{
	static constexpr int value =
			find_type_in_list<tags::is_clockwise, Others...>::value ?
					1 :
					(find_type_in_list<tags::is_unordered, Others...>::value ?
							0 : -1);
};
template<typename PrimitiveType, typename ...Others>
struct closure<model::Chains<PrimitiveType, Others...>>
{
	static constexpr int value =
			find_type_in_list<tags::is_closured, Others...>::value ?
					1 :
					(find_type_in_list<tags::is_unordered, Others...>::value ?
							0 : -1);
};
}  // namespace traits

}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_CHAINS_H_ */
