/**
 * @file boost_gemetry_adapted.h
 *
 * @date 2015-6-3
 * @author salmon
 */

#ifndef CORE_GEOMETRY_BOOST_GEOMETRY_ADAPTED_H_
#define CORE_GEOMETRY_BOOST_GEOMETRY_ADAPTED_H_

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>

#include "primitive.h"
#include "chains.h"
#include "model.h"
#include "CoordinateSystem.h"
namespace simpla
{
namespace geometry
{
namespace bg = boost::geometry;

using boost::geometry::append;

using boost::geometry::within;
using boost::geometry::disjoint;

using boost::geometry::distance;
using boost::geometry::length;
using boost::geometry::perimeter;

template<typename TGeo, typename ...Others> typename std::enable_if<
		traits::dimension<typename traits::coordinate_system<TGeo>::type>::value
				== 2, bool>::type intersection(TGeo const & geo,
		Others &&...others)
{
	boost::geometry::intersection(geo, std::forward<Others>(others)...);
}

template<typename TGeo, typename ...Others> typename std::enable_if<
		traits::dimension<typename traits::coordinate_system<TGeo>::type>::value
				!= 2, bool>::type intersection(TGeo const & geo,
		Others &&...others)
{
	return false;
}

template<typename TGeo, typename ...Others> typename std::enable_if<
		traits::dimension<typename traits::coordinate_system<TGeo>::type>::value
				== 2, bool>::type intersects(TGeo const & geo,
		Others &&...others)
{
	boost::geometry::intersection(geo, std::forward<Others>(others)...);
}

template<typename TGeo, typename ...Others> typename std::enable_if<
		traits::dimension<typename traits::coordinate_system<TGeo>::type>::value
				!= 2, bool>::type intersects(TGeo const & geo,
		Others &&...others)
{
	return false;
}

namespace detail
{

template<typename CS>
geometry::Point<CS> cross_product(const geometry::Point<CS>& p1,
		const geometry::Point<CS>& p2)
{
	double x = bg::get<0>(p1);
	double y = bg::get<1>(p1);
	double z = bg::get<2>(p1);
	double u = bg::get<0>(p2);
	double v = bg::get<1>(p2);
	double w = bg::get<2>(p2);
	return geometry::Point<CS>( { y * w - z * v, z * u - x * w, x * v - y * u });
}
template<typename CS>
geometry::Point<CS> cross_product(const bg::model::segment<geometry::Point<CS>>& p1,
		const bg::model::segment<geometry::Point<CS>>& p2)
{
	geometry::Point<CS> v1(p1.second);
	geometry::Point<CS> v2(p2.second);
	bg::subtract_point(v1, p1.first);
	bg::subtract_point(v2, p2.first);

	return cross_product(v1, v2);
}

template<typename CS>
auto area(geometry::Polyline<CS, tags::is_closed> const & polygon)
->decltype(std::declval<typename traits::coordinate_type<CS>::type>()*
		std::declval<typename traits::coordinate_type<CS>::type>())
{
	if (polygon.size() < 3)
		return 0;
	bg::model::segment<geometry::Point<CS>> v1(polygon[1], polygon[0]);
	bg::model::segment<geometry::Point<CS>> v2(polygon[2], polygon[0]);
	// Compute the cross product for the first pair of points, to handle
	// shapes that are not convex.
	geometry::Point<CS> n1 = cross_product(v1, v2);
	double normSquared = bg::dot_product(n1, n1);
	if (normSquared > 0)
	{
		bg::multiply_value(n1, 1.0 / std::sqrt(normSquared));
	}
	// sum signed areas of triangles
	double result = 0.0;
	for (size_t i = 1; i < polygon.size(); ++i)
	{
		bg::model::segment<geometry::Point<CS> > v1(polygon[0], polygon[i - 1]);
		bg::model::segment<geometry::Point<CS> > v2(polygon[0], polygon[i]);

		result += bg::dot_product(cross_product(v1, v2), n1);
	}
	result *= 0.5;
	return (result);
}
}  // namespace detail

template<typename CS> auto area(
		geometry::Polygon<CS> const & geo)->typename std::enable_if<
		traits::dimension<CS>::value == 2, Real>::type
{
	boost::geometry::area(geo);
}

template<typename CS> auto area(
		geometry::Polygon<CS> const & geo)->typename std::enable_if<
		traits::dimension<CS>::value != 2, Real>::type
{
	Real res = detail::area(geo.outer());
	for (auto const & r : geo.inners())
	{
		res -= detail::area(r);
	}

	return res;

}
}  // namespace geometry
}  // namespace simpla

namespace boost
{
namespace geometry
{
namespace traits
{
namespace sg = simpla::geometry;
namespace sgm = simpla:: engine::Model;
namespace sgcs = simpla::geometry::coordinate_system;

template<typename CS, typename TAG>
struct tag<sgm::Primitive<0, CS, TAG> >
{
	typedef point_tag type;
};

template<size_t N, typename CS, typename TAG>
struct coordinate_type<sgm::Primitive<N, CS, TAG> >
{
	typedef typename sg::traits::coordinate_type<CS>::type type;

};

template<size_t N, typename CS, typename TAG>
struct dimension<sgm::Primitive<N, CS, TAG>> : boost::mpl::int_<
		sg::traits::dimension<CS>::value>
{
};
template<size_t N, typename CS, typename TAG, std::size_t M>
struct access<sgm::Primitive<N, CS, TAG>, M>
{

	typedef sgm::Primitive<N, CS, TAG> Geo;

	typedef typename std::remove_reference<
			decltype((simpla::traits::get<M>( (std::declval<Geo>()))))>::type value_type;

	static inline value_type const &get(sgm::Primitive<N, CS, TAG>const& point)
	{
		return simpla::traits::get<M>(point);
	}

	template<typename T>
	static inline void set(sgm::Primitive<N, CS, TAG>& point, T const& value)
	{
		simpla::traits::get<M>(point) = static_cast<value_type>(value);
	}
};

template<size_t N, typename CS, typename TAG, size_t Index, size_t M>
struct indexed_access<sgm::Primitive<N, CS, TAG>, Index, M>
{
	typedef sgm::Primitive<N, CS, TAG> Geo;

	typedef typename std::remove_reference<
			decltype((simpla::traits::get<M,Index>( (std::declval<Geo>()))))>::type value_type;

	static inline value_type const & get(sgm::Primitive<N, CS, TAG> const& b)
	{
		return simpla::traits::get<M>(simpla::traits::get<Index>(b));
	}

	template<typename T>
	static inline void set(sgm::Primitive<N, CS, TAG>& b, T const& value)
	{
		simpla::traits::get<M, Index>(b) = static_cast<value_type>(value);
	}
};

template<size_t M, size_t N, typename TAG>
struct coordinate_system<sgm::Primitive<N, sgcs::Cartesian<M>, TAG> >
{
	typedef cs::cartesian type;
};

template<size_t N, typename TAG>
struct coordinate_system<sgm::Primitive<N, sgcs::Spherical, TAG> >

{
	typedef cs::spherical<radian> type;
};

template<size_t N, typename TAG>
struct coordinate_system<sgm::Primitive<N, sgcs::Polar, TAG> >
{
	typedef cs::spherical<radian> type;
};

//*******************************************************************
// Line Segment

template<typename CS, typename TAG>
struct tag<sgm::Primitive<1, CS, TAG> >
{
	typedef segment_tag type;
};
template<typename CS>
struct point_type<sgm::Primitive<1, CS, sg::tags::simplex> >
{
	typedef typename sgm::Primitive<0, CS, sg::tags::simplex> type;
};
//template<typename CS, std::size_t Dimension>
//struct indexed_access<sgm::Primitive<1, CS, sg::tags::simplex>, 0, Dimension>
//{
//	typedef sgm::Primitive<1, CS, sg::tags::simplex> segment_type;
//	typedef typename sg::traits::coordinate_type<CS>::value_type_info coordinate_type;
//
//	static inline coordinate_type Serialize(segment_type const& s)
//	{
//		return geometry::get<Dimension>(simpla::traits::Serialize<0>(s));
//	}
//
//	static inline void SetValue(segment_type& s, coordinate_type const& entity)
//	{
//		geometry::SetValue<Dimension>(simpla::traits::Serialize<0>(s), entity);
//	}
//};
//
//template<typename CS, std::size_t Dimension>
//struct indexed_access<sgm::Primitive<1, CS, sg::tags::simplex>, 1, Dimension>
//{
//	typedef sgm::Primitive<1, CS, sg::tags::simplex> segment_type;
//	typedef typename sg::traits::coordinate_type<CS>::value_type_info coordinate_type;
//
//	static inline coordinate_type Serialize(segment_type const& s)
//	{
//		return geometry::get<Dimension>(simpla::traits::Serialize<1>(s));
//	}
//
//	static inline void SetValue(segment_type& s, coordinate_type const& entity)
//	{
//		geometry::SetValue<Dimension>(simpla::traits::Serialize<1>(s), entity);
//	}
//};

//*******************************************************************
// RectMesh

template<typename CS>
struct tag<sgm::Box<CS>>
{
	typedef box_tag type;
};

template<typename CS>
struct point_type<sgm::Box<CS>>
{
	typedef sgm::Point<CS> type;
};

template<typename CS, size_t I, size_t Dimension>
struct indexed_access<sgm::Box<CS>, I, Dimension>
{

	static inline auto get(sgm::Box<CS> const& b)
	AUTO_RETURN(geometry::get<Dimension>(b[I]))

	template<typename T>
	static inline void set(sgm::Box<CS>& b, T const& value)
	{
		geometry::set<Dimension>(b[I], value);
	}
};
//********************************************************************
// ring

template<typename CS, typename ...Others>
struct tag<sgm::Polyline<CS, Others ...> >
{
	typedef ring_tag type;
};

template<typename CS, typename ...Others>
struct point_order<sgm::Polyline<CS, Others ...> >
{
	static const order_selector value =
			boost::geometry::order_selector::clockwise;
};

template<typename CS, typename ...Others>
struct closure<sgm::Polyline<CS, Others ...> >
{
	static constexpr closure_selector value =
			(simpla::mpl::find_type_in_list<
					simpla::geometry::tags::is_clockwise, Others...>::value) ?
					(boost::geometry::closure_selector::closed) :
					(boost::geometry::closure_selector::open);
	;
};

//********************************************************************
// Polygon

template<typename CS>
struct tag<sgm::Polygon<CS> >
{
	typedef polygon_tag type;
};

template<typename CS>
struct ring_const_type<sgm::Polygon<CS>>
{
	typedef typename sgm::Polygon<CS>::ring_type const& type;
};

template<typename CS>
struct ring_mutable_type<sgm::Polygon<CS>>
{
	typedef typename sgm::Polygon<CS>::ring_type& type;
};

template<typename CS>
struct interior_const_type<sgm::Polygon<CS>>
{
	typedef typename sgm::Polygon<CS>::inner_container_type const& type;
};

template<typename CS>
struct interior_mutable_type<sgm::Polygon<CS>>
{
	typedef typename sgm::Polygon<CS>::inner_container_type& type;
};

template<typename CS>
struct exterior_ring<sgm::Polygon<CS>>
{
	typedef sgm::Polygon<CS> polygon_type;

	static inline typename polygon_type::ring_type& get(polygon_type& p)
	{
		return p.outer();
	}

	static inline typename polygon_type::ring_type const& get(
			polygon_type const& p)
	{
		return p.outer();
	}
};

template<typename CS>
struct interior_rings<sgm::Polygon<CS>>
{
	typedef sgm::Polygon<CS> polygon_type;

	static inline typename polygon_type::inner_container_type& get(
			polygon_type& p)
	{
		return p.inners();
	}

	static inline typename polygon_type::inner_container_type const& get(
			polygon_type const& p)
	{
		return p.inners();
	}
};

} // namespace traits
} // namespace geometry
} // namespace boost
#endif // CORE_GEOMETRY_BOOST_GEOMETRY_ADAPTED_H_
