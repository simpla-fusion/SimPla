/**
 * @file manifold.h
 *
 * @date 2015年6月4日
 * @author salmon
 */

#ifndef CORE_GEOMETRY_MANIFOLD_H_
#define CORE_GEOMETRY_MANIFOLD_H_

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
template<size_t N, typename CoordinateSystem, typename Tag, typename ...> struct Element;

#define DEF_NTUPLE_OBJECT(_COORD_SYS_)                                                       \
 typedef typename simpla::geometry::coordinate_system::traits::coordinate_type<              \
		 _COORD_SYS_>::type value_type;                                                      \
                                                                                             \
 static const size_t ndims =	simpla::geometry::coordinate_system::traits::dimension<      \
         _COORD_SYS_>::value;                                                                \
                                                                                             \
 nTuple<value_type, ndims> m_data_;                                                          \
                                                                                             \
 inline operator nTuple<value_type, ndims>(){ return m_data_; }                              \
                                                                                             \
 inline value_type & operator [](size_t n){ return m_data_[n]; }                             \
 inline value_type const & operator [](size_t n) const{ return m_data_[n]; }                 \
 template<size_t N> inline value_type & get(){return m_data_[N]; }                           \
 template<size_t N> inline constexpr value_type const& get() const                           \
 { return m_data_[N]; }                                                                      \
                                                                                             \

/**
 * @brief Point topological 0-dimensional 'geometric primitive' ,
 *   representing a position
 * @note The boundary of a point is the empty set. [ISO 19107]
 */
template<typename CoordinateSystem, typename ...Others>
struct Element<0, CoordinateSystem, Others...>
{
	DEF_NTUPLE_OBJECT(CoordinateSystem);
};

/**
 * @brief Vector In geometry, Vector represents the first derivative of 'curve',
 * call element $v\in T_P M$ 'vectors' at point $P\in M$; $T_P M$ is the 'tagent space'
 * at the point $P$
 * In code,Vector is the difference type of Point Vector = Point - Point
 */
template<typename CoordinateSystem>
struct Vector
{
	DEF_NTUPLE_OBJECT(CoordinateSystem);
};

/**
 * @brief CoVector is a linear map from  'vector space'
 *
 */
template<typename CoordinateSystem>
struct CoVector
{
	DEF_NTUPLE_OBJECT
};

#undef DEF_NTUPLE_OBJECT

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

	template<size_t N>
	inline auto get()
	DECL_RET_TYPE (m_data_[N])

	template<size_t N>
	inline auto get() const
	DECL_RET_TYPE (m_data_[N])
};
namespace tags
{
struct is_simplex;
struct is_cube;

}  // namespace tags

namespace traits
{
template<typename > struct coordinate_system;
template<typename > struct dimension;
template<typename > struct vertex_type;
template<typename > struct number_of_vertices;
template<typename, typename > struct check_tag;

template<size_t Dimension, typename CoordinateSystem, typename ...Others>
struct coordinate_system<Element<Dimension, CoordinateSystem, Others...>>
{
	typedef CoordinateSystem type;
};

template<size_t Dimension, typename ...Others>
struct dimension<Element<Dimension, Others...>>
{
	static constexpr size_t value = Dimension;
};
template<size_t Dimension, typename CoordinateSystem, typename ...Others>
struct vertex_type<Element<Dimension, CoordinateSystem, Others...>>
{
	typedef Element<0, CoordinateSystem, Others...> vertex_type;
};
template<typename ...Others>
struct number_of_vertices<Element<0, Others...>>
{
	static constexpr size_t value = 1;
};

template<size_t Dimension, typename ...Others>
struct number_of_vertices<Element<Dimension, Others...>>
{
	typedef Element<Dimension, Others...> type;

	static constexpr size_t value =
			check_tag<tags::is_simplex, type>::value ?
					(Dimension + 1)

					:
					(check_tag<tags::is_cube, type>::value ?
							(2
									* number_of_vertices<
											Element<Dimension - 1, Others...>>::value)

							:
							(0)

					);
};

template<typename T, size_t Dimension, typename ...Others>
struct check_tag<T, Element<Dimension, Others...>>
{
	static constexpr bool value = find_type_in_list<T, Others...>::value;
};
}  // namespace traits

//************************************************************************
//************************************************************************
//************************************************************************
//************************************************************************
//************************************************************************

template<typename ...> struct Chains;

template<typename CoordinateSystem, typename ...Others>
using Point = Element< 0,CoordinateSystem, Others...>;

template<typename CoordinateSystem, typename ...Others>
using LineSegment = Element< 1,CoordinateSystem, Others...>;

template<typename CoordinateSystem, typename ...Others>
using Pixel = Element< 2,CoordinateSystem, Others...>;

template<typename CoordinateSystem, typename ...Others>
using Voxel = Element< 3,CoordinateSystem, Others...>;

template<typename ...Others, typename ...Others2>
using PointSet=Chains<Element<0, Others...>,Others2...>;

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
template<typename ...Others, typename ...Others2>
using Curve=Chains<Element<1, Others...>,Others2...>;

/**
 * @brief Surface
 * topological 2-dimensional  geometric primitive (4.15),
 * locally representing a continuous image of a region of a plane
 * @note The boundary of a surface is the set of oriented, closed curves
 *  that delineate the limits of the surface.
 *
 */
template<typename ...Others, typename ...Others2>
using Surface=Chains<Element<2, Others...>,Others2...>;

/**
 * @brief Solids
 */

template<typename ...Others, typename ...Others2>
using Solids=Chains<Element<3, Others...>,Others2...>;
}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_MANIFOLD_H_ */
