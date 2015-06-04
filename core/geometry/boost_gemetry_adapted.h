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

template<typename TPoint> using Polygon= boost::geometry::model::polygon<TPoint>;

template<typename TPoint> using Box= boost::geometry::model::box<TPoint>;

template<typename TPoint> using LineSegment= boost::geometry::model::segment<TPoint>;

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

template<typename CS, typename TAG>
struct tag<simpla::geometry::Primitive<0, CS, TAG> >
{
	typedef point_tag type;
};

template<size_t N, typename CS, typename TAG>
struct coordinate_type<simpla::geometry::Primitive<N, CS, TAG> >
{
	typedef typename simpla::geometry::coordinate_system::traits::coordinate_type<
			CS>::type type;

};

template<size_t N, typename CS, typename TAG>
struct dimension<simpla::geometry::Primitive<N, CS, TAG>> : boost::mpl::int_<
		simpla::geometry::coordinate_system::traits::dimension<CS>::value>
{
};
template<typename CS, typename TAG, std::size_t Dimension>
struct access<simpla::geometry::Primitive<0, CS, TAG>, Dimension>
{
	typedef typename coordinate_type<simpla::geometry::Primitive<0, CS, TAG>>::type value_type;

	static inline value_type const &get(
			simpla::geometry::Primitive<0, CS, TAG>const& point)
	{
		return std::get<Dimension>(point);
	}

	static inline void set(simpla::geometry::Primitive<0, CS, TAG>& point,
			value_type const& value)
	{
		std::get<Dimension>(point) = value;
	}
};
template<size_t M, size_t N, typename TAG>
struct coordinate_system<
		simpla::geometry::Primitive<N,
				simpla::geometry::coordinate_system::Cartesian<M>, TAG> >
{
	typedef cs::cartesian type;
};

template<size_t N, typename TAG>
struct coordinate_system<
		simpla::geometry::Primitive<N,
				simpla::geometry::coordinate_system::Spherical, TAG> >

{
	typedef cs::spherical<radian> type;
};

template<size_t N, typename TAG>
struct coordinate_system<
		simpla::geometry::Primitive<N,
				simpla::geometry::coordinate_system::Polar, TAG> >
{
	typedef cs::spherical<radian> type;
};
} // namespace traits
#endif // DOXYGEN_NO_TRAITS_SPECIALIZATIONS
}
}
#endif // CORE_GEOMETRY_BOOST_GEMETRY_ADAPTED_H_
