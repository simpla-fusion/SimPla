/**
 * @file primitive.h
 *
 * @date 2015年6月4日
 * @author salmon
 */

#ifndef CORE_GEOMETRY_PRIMITIVE_H_
#define CORE_GEOMETRY_PRIMITIVE_H_
#include "../gtl/type_traits.h"
#include "coordinate_system.h"
/**
 *  @ref OpenGIS® Implementation Standard for Geographic information
 *  - Simple feature access -architecture Part 1: Common architecture
 *  @ref boost::geometry GGL
 */
namespace simpla
{

namespace geometry
{

/**
 *  @brief Manifold or geometric primitive , geometric primitive
 *  representing a single, connected, homogeneous element of space
 */
/**
 * @brief Element topological n-dimensional 'geometric primitive' ,
 * Element<0> is point (simplex<0>)
 * Element<1> is a segment of straight line (simplex<1>) or curve, has to end-point
 * Element<2> is triangle (simplex<2>) , rectangle, etc...
 * Element<3> is tetrahedron (simplex<3>) , cube , etc...
 */
template<size_t Dimension, typename ...> struct Primitive;

namespace tags
{
struct simplex;
struct cube;

}  // namespace tags

namespace traits
{
template<typename > struct coordinate_system;
template<typename > struct dimension;
template<typename > struct vertex_type;
template<typename > struct number_of_vertices;

template<typename > struct is_chains;
template<typename > struct is_primitive;

template<size_t Dimension, typename ...Others>
struct is_primitive<Primitive<Dimension, Others...>>
{
	static constexpr bool value = true;
};

template<size_t Dimension, typename ...Others>
struct is_chains<Primitive<Dimension, Others...>>
{
	static constexpr bool value = false;
};

template<size_t Dimension, typename CoordinateSystem, typename Tag>
struct coordinate_system<Primitive<Dimension, CoordinateSystem, Tag>>
{
	typedef CoordinateSystem type;
};

template<size_t Dimension, typename CoordinateSystem, typename Tag>
struct dimension<Primitive<Dimension, CoordinateSystem, Tag>>
{
	static constexpr size_t value = Dimension;
};
template<size_t Dimension, typename CoordinateSystem, typename Tag>
struct vertex_type<Primitive<Dimension, CoordinateSystem, Tag>>
{
	typedef Primitive<0, CoordinateSystem, tags::simplex> type;
};
template<typename ...Others>
struct number_of_vertices<Primitive<0, Others...>>
{
	static constexpr size_t value = 1;
};

template<size_t Dimension, typename CoordinateSystem, typename Tag>
struct number_of_vertices<Primitive<Dimension, CoordinateSystem, Tag>>
{

	static constexpr size_t value =
			std::is_same<tags::simplex, Tag>::value ?
					(Dimension + 1)

					:
					(std::is_same<tags::cube, Tag>::value ?
							(2
									* number_of_vertices<
											Primitive<Dimension - 1,
													CoordinateSystem, Tag>>::value)

							:
							(0)

					);
};

} // namespace traits

#define DEF_NTUPLE_OBJECT(_COORD_SYS_,_T_,_NUM_)                                      \
 nTuple<_T_, _NUM_> m_data_;                                                          \
 inline operator nTuple<_T_, _NUM_>(){ return m_data_; }                              \
 inline _T_ & operator [](size_t n){ return m_data_[n]; }                             \
 inline _T_ const & operator [](size_t n) const{ return m_data_[n]; }                 \
 template<size_t N> inline _T_ & get(){return m_data_[N]; }                           \
 template<size_t N> inline constexpr _T_ const& get() const  { return m_data_[N]; }

/**
 * @brief Point topological 0-dimensional 'geometric primitive' ,
 *   representing a position
 * @note The boundary of a point is the empty set. [ISO 19107]
 */
template<typename CoordinateSystem, typename Tag>
struct Primitive<0, CoordinateSystem, Tag>
{
	typedef typename simpla::geometry::coordinate_system::traits::coordinate_type<
			CoordinateSystem>::type value_type;

	static const size_t ndims =
			simpla::geometry::coordinate_system::traits::dimension<
					CoordinateSystem>::value;

	DEF_NTUPLE_OBJECT(CoordinateSystem,value_type,ndims);
};
template<typename CoordinateSystem>
using Point= Primitive<0, CoordinateSystem, tags::simplex>;

/**
 * @brief Vector In geometry, Vector represents the first derivative of 'curve',
 * call element $v\in T_P M$ 'vectors' at point $P\in M$; $T_P M$ is the 'tagent space'
 * at the point $P$
 * In code,Vector is the difference type of Point Vector = Point - Point
 */
template<typename CoordinateSystem>
struct Vector
{
	typedef typename simpla::geometry::coordinate_system::traits::coordinate_type<
			CoordinateSystem>::type value_type;

	static const size_t ndims =
			simpla::geometry::coordinate_system::traits::dimension<
					CoordinateSystem>::value;

	DEF_NTUPLE_OBJECT(CoordinateSystem, value_type, ndims);

};

/**
 * @brief CoVector is a linear map from  'vector space'
 *
 */
template<typename CoordinateSystem>
struct CoVector
{
typedef typename simpla::geometry::coordinate_system::traits::coordinate_type<
		CoordinateSystem>::type value_type;

static const size_t ndims =
		simpla::geometry::coordinate_system::traits::dimension<CoordinateSystem>::value;

DEF_NTUPLE_OBJECT(CoordinateSystem,value_type,ndims);
};

/**
 * THIS is INCOMPLETE!!!
 */
template<typename CoordinateSystem, typename CoordinateType, size_t ...N>
struct Tensor
{
	nTuple<CoordinateType, N...> m_data_;

	inline operator nTuple<CoordinateType, N...>()
	{
		return m_data_;
	}

	inline auto operator [](size_t n)
	DECL_RET_TYPE (m_data_[n])

	inline auto operator [](size_t n) const
	DECL_RET_TYPE (m_data_[n])

	template<size_t M>
	inline auto get()
	DECL_RET_TYPE (m_data_[M])

	template<size_t M>
	inline auto get() const
	DECL_RET_TYPE (m_data_[M])
};

template<size_t Dimension, typename CoordinateSystem, typename Tag>
struct Primitive<Dimension, CoordinateSystem, Tag>
{
	typedef Primitive<Dimension, CoordinateSystem, Tag> this_type;
	typedef typename traits::vertex_type<this_type>::type vertex_type;
	static constexpr size_t num_of_vertices = traits::number_of_vertices<
			this_type>::value;

	DEF_NTUPLE_OBJECT(CoordinateSystem,vertex_type,num_of_vertices);
};
#undef DEF_NTUPLE_OBJECT

//template<typename CoordinateSystem>
//using Point = Primitive< 0,CoordinateSystem, tags::simplex>;

template<typename CoordinateSystem, typename Tag>
using Line = Primitive< 1,CoordinateSystem, tags::simplex >;

template<typename CoordinateSystem, typename Tag>
using Triangle = Primitive< 2,CoordinateSystem, tags::simplex >;

template<typename CoordinateSystem, typename Tag>
using Tetrahedron = Primitive< 2,CoordinateSystem, tags::simplex >;

template<typename CoordinateSystem, typename Tag>
using Rectangle = Primitive< 2,CoordinateSystem, tags::cube >;

template<typename CoordinateSystem, typename Tag>
using Cube = Primitive< 3,CoordinateSystem, tags::cube >;

//************************************************************************
//************************************************************************
//************************************************************************
//************************************************************************

}// namespace geometry
}  // namespace simpla

namespace std
{

template<size_t N, size_t Dimension, typename ... Others>
auto get(simpla::geometry::Primitive<Dimension, Others...> & obj)
DECL_RET_TYPE((obj[N]))

template<size_t N, size_t Dimension, typename ...Others>
auto get(simpla::geometry::Primitive<Dimension, Others...> const & obj)
DECL_RET_TYPE((obj[N]))

}  // namespace std
#endif /* CORE_GEOMETRY_PRIMITIVE_H_ */
