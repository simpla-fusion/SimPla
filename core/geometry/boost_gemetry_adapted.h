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

namespace boost
{
namespace geometry
{

#ifndef DOXYGEN_NO_TRAITS_SPECIALIZATIONS
namespace traits
{

template<typename T1, size_t N>
struct tag<simpla::nTuple<T1, N> >
{
	typedef point_tag type;
};

template<typename T1, size_t N>
struct coordinate_type<simpla::nTuple<T1, N> >
{
	typedef T1 type;
};

template<typename T1, size_t N>
struct dimension<simpla::nTuple<T1, N>> : boost::mpl::int_<N>
{
};

template<typename T1, size_t N, std::size_t Dimension>
struct access<simpla::nTuple<T1, N>, Dimension>
{
	static inline T1 get(simpla::nTuple<T1, N> const& point)
	{
		return point[Dimension];
	}

	static inline void set(simpla::nTuple<T1, N>& point, T1 const& value)
	{
		point[Dimension] = value;
	}
};
template<typename T1, size_t N>
struct coordinate_system<simpla::nTuple<T1, N> >
{
	typedef cs::cartesian type;
};
} // namespace traits
#endif // DOXYGEN_NO_TRAITS_SPECIALIZATIONS
}  // namespace geometry
}  // namespace boost

//namespace cs
//{
//template<typename DegreeOrRadian>
//struct cylindrical
//{
//	typedef DegreeOrRadian units;
//};
//
//}  // namespace cs

//BOOST_GEOMETRY_REGISTER_POINT_2D(simpla::geometry::SphericalPoint2, double,
//		cs::spherical<radian>, r, theta)
//
//BOOST_GEOMETRY_REGISTER_POINT_3D(simpla::geometry::SphericalPoint3, double,
//		cs::cartesian, r, theta, phi)
//
//BOOST_GEOMETRY_REGISTER_POINT_2D(simpla::geometry::CylindricalPoint2, double,
//		cs::cylindrical<radian>, r, theta)
//
//BOOST_GEOMETRY_REGISTER_POINT_3D(simpla::geometry::CylindricalPoint3, double,
//		cs::cylindrical<radian>, r, theta, z)

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
using boost::geometry::envelope;

using boost::geometry::distance;
using boost::geometry::area;
using boost::geometry::length;
using boost::geometry::perimeter;

} // namespace geometry
}  // namespace simpla

#endif // CORE_GEOMETRY_BOOST_GEMETRY_ADAPTED_H_
