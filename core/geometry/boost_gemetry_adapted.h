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
#include "chains.h"
#include "model.h"

namespace simpla
{
namespace geometry
{
namespace model
{

template<size_t N, typename CS, typename TAG = tags::simplex>
using Polygon=boost::geometry::model::polygon<Primitive<N,CS,TAG>>;

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
namespace traits
{
namespace sg = simpla::geometry;
namespace sgm = simpla::geometry::model;
namespace sgcs = simpla::geometry::coordinate_system;

template<typename CS, typename TAG>
struct tag<sgm::Primitive<0, CS, TAG> >
{
	typedef point_tag type;
};

template<size_t N, typename CS, typename TAG>
struct coordinate_type<sgm::Primitive<N, CS, TAG> >
{
	typedef typename sgcs::traits::coordinate_type<CS>::type type;

};

template<size_t N, typename CS, typename TAG>
struct dimension<sgm::Primitive<N, CS, TAG>> : boost::mpl::int_<
		sgcs::traits::dimension<CS>::value>
{
};
template<typename CS, typename TAG, std::size_t Dimension>
struct access<sgm::Primitive<0, CS, TAG>, Dimension>
{
	typedef typename coordinate_type<sgm::Primitive<0, CS, TAG>>::type value_type;

	static inline value_type const &get(sgm::Primitive<0, CS, TAG>const& point)
	{
		return std::get<Dimension>(point);
	}

	static inline void set(sgm::Primitive<0, CS, TAG>& point,
			value_type const& value)
	{
		std::get<Dimension>(point) = value;
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

template<typename CS>
struct tag<sgm::Primitive<1, CS, sg::tags::simplex> >
{
	typedef segment_tag type;
};
template<typename CS>
struct point_type<sgm::Primitive<1, CS, sg::tags::simplex> >
{
	typedef typename sgm::Primitive<0, CS, sg::tags::simplex> type;
};
template<typename CS, std::size_t Dimension>
struct indexed_access<sgm::Primitive<1, CS, sg::tags::simplex>, 0, Dimension>
{
	typedef sgm::Primitive<1, CS, sg::tags::simplex> segment_type;
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
struct indexed_access<sgm::Primitive<1, CS, sg::tags::simplex>, 1, Dimension>
{
	typedef sgm::Primitive<1, CS, sg::tags::simplex> segment_type;
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
struct tag<sgm::Primitive<N, CS, sg::tags::box>>
{
	typedef box_tag type;
};

template<size_t N, typename CS>
struct point_type<sgm::Primitive<N, CS, sg::tags::box>>
{
	typedef typename sg::traits::point_type<sgm::Primitive<N, CS, sg::tags::box>>::type type;
};

template<size_t N, typename CS, std::size_t Dimension>
struct indexed_access<sgm::Primitive<N, CS, sg::tags::box>, min_corner,
		Dimension>
{
	typedef typename geometry::coordinate_type<CS>::type coordinate_type;

	static inline coordinate_type get(
			sgm::Primitive<N, CS, sg::tags::box> const& b)
	{
		return geometry::get<Dimension>(std::get<0>(b));
	}

	static inline void set(sgm::Primitive<N, CS, sg::tags::box>& b,
			coordinate_type const& value)
	{
		geometry::set<Dimension>(std::get<0>(b), value);
	}
};

template<size_t N, typename CS, std::size_t Dimension>
struct indexed_access<sgm::Primitive<N, CS, sg::tags::box>, max_corner,
		Dimension>
{
	typedef typename geometry::coordinate_type<CS>::type coordinate_type;

	static inline coordinate_type get(
			sgm::Primitive<N, CS, sg::tags::box> const& b)
	{
		return geometry::get<Dimension>(std::get<1>(b));
	}

	static inline void set(sgm::Primitive<N, CS, sg::tags::box>& b,
			coordinate_type const& value)
	{
		geometry::set<Dimension>(std::get<1>(b), value);
	}
};
//********************************************************************
// ring

template<size_t N, typename CS, typename TAG, typename ...Others>
struct tag<sgm::Chains<sgm::Primitive<N, CS, TAG>, Others ...> >
{
	typedef ring_tag type;
};

template<size_t N, typename CS, typename TAG, typename ...Others>
struct point_order<sgm::Chains<sgm::Primitive<N, CS, TAG>, Others ...> >
{
	static const order_selector value = simpla::geometry::traits::point_order<
			sgm::Chains<sgm::Primitive<N, CS, TAG>, Others ...>>::value;
	;
};

template<size_t N, typename CS, typename TAG, typename ...Others>
struct closure<sgm::Chains<sgm::Primitive<N, CS, TAG>, Others ...> >
{
	static const closure_selector value = sg::traits::closure<
			sgm::Chains<sgm::Primitive<N, CS, TAG>, Others ...>>::value;
};

//********************************************************************
// Polygon

template<size_t N, typename CS, typename TAG, typename ...Others>
struct tag<sgm::expPolygon<N, CS, TAG, Others ...> >
{
	typedef polygon_tag type;
};

template<size_t N, typename CS, typename TAG, typename ...Others>
struct ring_const_type<sgm::expPolygon<N, CS, TAG, Others ...>>
{
	typedef typename sgm::expPolygon<N, CS, TAG, Others ...>::ring_type const& type;
};

template<size_t N, typename CS, typename TAG, typename ...Others>
struct ring_mutable_type<sgm::expPolygon<N, CS, TAG, Others ...>>
{
	typedef typename sgm::expPolygon<N, CS, TAG, Others ...>::ring_type& type;
};

template<size_t N, typename CS, typename TAG, typename ...Others>
struct interior_const_type<sgm::expPolygon<N, CS, TAG, Others ...>>
{
	typedef typename sgm::expPolygon<N, CS, TAG, Others ...>::inner_container_type const& type;
};

template<size_t N, typename CS, typename TAG, typename ...Others>
struct interior_mutable_type<sgm::expPolygon<N, CS, TAG, Others ...>>
{
	typedef typename sgm::expPolygon<N, CS, TAG, Others ...>::inner_container_type& type;
};

template<size_t N, typename CS, typename TAG, typename ...Others>
struct exterior_ring<sgm::expPolygon<N, CS, TAG, Others ...>>
{
	typedef sgm::expPolygon<N, CS, TAG, Others ...> polygon_type;

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

template<size_t N, typename CS, typename TAG, typename ...Others>
struct interior_rings<sgm::expPolygon<N, CS, TAG, Others ...>>
{
	typedef sgm::expPolygon<N, CS, TAG, Others ...> polygon_type;

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
#endif // CORE_GEOMETRY_BOOST_GEMETRY_ADAPTED_H_
