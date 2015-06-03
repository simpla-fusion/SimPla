/**
 * @file boost_gemetry_adapted.h
 *
 * @date 2015年6月3日
 * @author salmon
 */

#ifndef CORE_GEOMETRY_BOOST_GEMETRY_ADAPTED_H_
#define CORE_GEOMETRY_BOOST_GEMETRY_ADAPTED_H_

#include <cstddef>

#include <boost/tuple/tuple.hpp>

#include <boost/geometry/core/coordinate_dimension.hpp>
#include <boost/geometry/core/coordinate_type.hpp>
#include <boost/geometry/core/point_type.hpp>
#include <boost/geometry/core/tags.hpp>

#include "../gtl/ntuple.h"

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
		point[Dimension]= value;
	}
};

} // namespace traits
#endif // DOXYGEN_NO_TRAITS_SPECIALIZATIONS

}
} // namespace boost::geometry

// Convenience registration macro to bind boost::tuple to a CS
#define BOOST_GEOMETRY_REGISTER_SIMPLA_NTUPLE_CS(CoordinateSystem) \
    namespace boost { namespace geometry { namespace traits { \
    template <typename T1, size_t N> \
    struct coordinate_system<simpla::nTuple<T1, N> > \
    { \
        typedef CoordinateSystem type; \
    }; \
    }}}

#endif /* CORE_GEOMETRY_BOOST_GEMETRY_ADAPTED_H_ */
