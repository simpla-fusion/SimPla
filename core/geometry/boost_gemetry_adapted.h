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
template<typename CS> using Polygon= boost::geometry::model::polygon< Point<CS> >;

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
namespace traits
{
namespace sg = simpla::geometry;
namespace sgcs = simpla::geometry::coordinate_system;
template<typename CS, typename TAG>
struct tag<sg::Primitive<0, CS, TAG> >
{
	typedef point_tag type;
};

template<size_t N, typename CS, typename TAG>
struct coordinate_type<sg::Primitive<N, CS, TAG> >
{
	typedef typename sgcs::traits::coordinate_type<CS>::type type;

};

template<size_t N, typename CS, typename TAG>
struct dimension<sg::Primitive<N, CS, TAG>> : boost::mpl::int_<
		sgcs::traits::dimension<CS>::value>
{
};
template<typename CS, typename TAG, std::size_t Dimension>
struct access<sg::Primitive<0, CS, TAG>, Dimension>
{
	typedef typename coordinate_type<sg::Primitive<0, CS, TAG>>::type value_type;

	static inline value_type const &get(sg::Primitive<0, CS, TAG>const& point)
	{
		return std::get<Dimension>(point);
	}

	static inline void set(sg::Primitive<0, CS, TAG>& point,
			value_type const& value)
	{
		std::get<Dimension>(point) = value;
	}
};
template<size_t M, size_t N, typename TAG>
struct coordinate_system<sg::Primitive<N, sgcs::Cartesian<M>, TAG> >
{
	typedef cs::cartesian type;
};

template<size_t N, typename TAG>
struct coordinate_system<sg::Primitive<N, sgcs::Spherical, TAG> >

{
	typedef cs::spherical<radian> type;
};

template<size_t N, typename TAG>
struct coordinate_system<sg::Primitive<N, sgcs::Polar, TAG> >
{
	typedef cs::spherical<radian> type;
};

//*******************************************************************
// Line Segment

template<typename CS>
struct tag<sg::Primitive<1, CS, sg::tags::simplex> >
{
	typedef segment_tag type;
};
template<typename CS>
struct point_type<sg::Primitive<1, CS, sg::tags::simplex> >
{
	typedef typename sg::traits::point_type<CS>::type type;
};
template<typename CS, std::size_t Dimension>
struct indexed_access<sg::Primitive<1, CS, sg::tags::simplex>, 0, Dimension>
{
	typedef sg::Primitive<1, CS, sg::tags::simplex> segment_type;
	typedef typename sgcs::traits::coordinate_type<CS>::type coordinate_type;

	static inline coordinate_type get(segment_type const& s)
	{
		return geometry::get<Dimension>(std::get<0>(s));
	}

	static inline void set(segment_type& s, coordinate_type const& value)
	{
		geometry::set<Dimension>(std::get<0>(s), value);
	}
};

template<typename CS, std::size_t Dimension>
struct indexed_access<sg::Primitive<1, CS, sg::tags::simplex>, 1, Dimension>
{
	typedef sg::Primitive<1, CS, sg::tags::simplex> segment_type;
	typedef typename sgcs::traits::coordinate_type<CS>::type coordinate_type;

	static inline coordinate_type get(segment_type const& s)
	{
		return geometry::get<Dimension>(std::get<1>(s));
	}

	static inline void set(segment_type& s, coordinate_type const& value)
	{
		geometry::set<Dimension>(std::get<1>(s), value);
	}
};
//*******************************************************************
// Box

template<size_t N, typename CS>
struct tag<sg::Primitive<N, CS, sg::tags::box>>
{
	typedef box_tag type;
};

template<size_t N, typename CS>
struct point_type<sg::Primitive<N, CS, sg::tags::box>>
{
	typedef typename sg::traits::point_type<sg::Primitive<N, CS, sg::tags::box>>::type type;
};

template<size_t N, typename CS, std::size_t Dimension>
struct indexed_access<sg::Primitive<N, CS, sg::tags::box>, min_corner, Dimension>
{
	typedef typename geometry::coordinate_type<CS>::type coordinate_type;

	static inline coordinate_type get(
			sg::Primitive<N, CS, sg::tags::box> const& b)
	{
		return geometry::get<Dimension>(std::get<0>(b));
	}

	static inline void set(sg::Primitive<N, CS, sg::tags::box>& b,
			coordinate_type const& value)
	{
		geometry::set<Dimension>(std::get<0>(b), value);
	}
};

template<size_t N, typename CS, std::size_t Dimension>
struct indexed_access<sg::Primitive<N, CS, sg::tags::box>, max_corner, Dimension>
{
	typedef typename geometry::coordinate_type<CS>::type coordinate_type;

	static inline coordinate_type get(
			sg::Primitive<N, CS, sg::tags::box> const& b)
	{
		return geometry::get<Dimension>(std::get<1>(b));
	}

	static inline void set(sg::Primitive<N, CS, sg::tags::box>& b,
			coordinate_type const& value)
	{
		geometry::set<Dimension>(std::get<1>(b), value);
	}
};
} // namespace traits
} // namespace geometry
} // namespace boost
#endif // CORE_GEOMETRY_BOOST_GEMETRY_ADAPTED_H_
