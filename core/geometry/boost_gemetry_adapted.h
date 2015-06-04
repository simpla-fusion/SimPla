/**
 * @file boost_gemetry_adapted.h
 *
 * @date 2015年6月3日
 * @author salmon
 */

#ifndef CORE_GEOMETRY_BOOST_GEMETRY_ADAPTED_H_
#define CORE_GEOMETRY_BOOST_GEMETRY_ADAPTED_H_

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>

#include "primitive.h"

namespace simpla
{
namespace geometry
{
namespace model
{

template<typename TPoint> using Polygon= boost::geometry::model::polygon<TPoint>;

template<typename TPoint> using Box= boost::geometry::model::box<TPoint>;

template<typename TPoint> using LineSegment= boost::geometry::model::segment<TPoint>;

}  // namespace model

using boost::geometry::append;
using boost::geometry::intersection;
using boost::geometry::intersects;
using boost::geometry::within;
using boost::geometry::disjoint;
using boost::geometry::dsv;

using boost::geometry::distance;
using boost::geometry::area;
using boost::geometry::length;
using boost::geometry::perimeter;

}  // namespace geometry
}  // namespace simpla

namespace boost
{
namespace geometry
{
#ifndef DOXYGEN_NO_TRAITS_SPECIALIZATIONS
namespace traits
{

template<typename T1, typename CS>
struct tag<simpla::geometry::model::Manifold<CS, T1, 0> >
{
	typedef point_tag type;
};

template<typename T1, typename CS, size_t N>
struct coordinate_type<simpla::geometry::model::Manifold<CS, T1, N> >
{
	typedef T1 type;
};

template<typename T1, typename CS, size_t N>
struct dimension<simpla::geometry::model::Manifold<CS, N, T1>> : boost::mpl::int_<
		simpla::geometry::coordinate_system::traits::dimension<CS>::value>
{
};
template<typename T1, typename CS, std::size_t Dimension>
struct access<simpla::geometry::model::Manifold<CS, T1, 0>, Dimension>
{
	static inline T1 const &get(
			simpla::geometry::model::Manifold<CS, T1, 0> const& point)
	{
		return point.template get<Dimension>();
	}

	static inline void set(simpla::geometry::model::Manifold<CS, T1, 0>& point,
			T1 const& value)
	{
		point.template get<Dimension>() = value;
	}
};
template<typename T1, size_t M, size_t N>
struct coordinate_system<
		simpla::geometry::model::Manifold<
				simpla::geometry::coordinate_system::Cartesian<M>, N, T1> >
{
	typedef cs::cartesian type;
};

template<typename T1, size_t N>
struct coordinate_system<
		simpla::geometry::model::Manifold<
				simpla::geometry::coordinate_system::Spherical, N, T1> >

{
	typedef cs::spherical<radian> type;
};

template<typename T1, size_t N>
struct coordinate_system<
		simpla::geometry::model::Manifold<
				simpla::geometry::coordinate_system::Polar, N, T1> >
{
	typedef cs::spherical<radian> type;
};
} // namespace traits
#endif // DOXYGEN_NO_TRAITS_SPECIALIZATIONS
}
}
#endif // CORE_GEOMETRY_BOOST_GEMETRY_ADAPTED_H_
