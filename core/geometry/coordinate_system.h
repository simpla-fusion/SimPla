/**
 * @file coordinate_system.h
 *
 * @date 2015-6-3
 * @author salmon
 */

#ifndef CORE_GEOMETRY_COORDINATE_SYSTEM_H_
#define CORE_GEOMETRY_COORDINATE_SYSTEM_H_

#include <stddef.h>
#include <cstdbool>
#include <type_traits>

#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"

namespace simpla
{
namespace geometry
{
template<typename, typename> struct map;
template<typename> struct mertic;

namespace traits
{

template<typename> struct dimension;
template<typename> struct coordinate_type;
template<typename> struct is_homogeneous;
template<typename> struct typename_as_string;
template<typename> struct ZAxis;
}  // namespace traits

namespace coordinate_system
{

struct Polar
{
};

template<size_t N, size_t ZAXIS = 2>
struct Cartesian
{
};

struct Spherical
{
};
template<int IPhiAxis = 2>
struct Cylindrical
{
};

struct Toroidal
{
};

struct MagneticFLux
{
};

} //namespace coordinate_system

namespace traits
{

template<typename CS>
struct scalar_type
{
	typedef Real type;
};
template<typename CS>
using scalar_type_t =typename scalar_type<CS>::type;

template<typename CS>
struct point_type
{
	typedef nTuple <Real, dimension<CS>::value> type;
};
template<typename CS>
using point_t=typename point_type<CS>::type;
template<typename CS>
using point_type_t=typename point_type<CS>::type;
template<typename CS>
struct vector_type
{
	typedef nTuple <Real, dimension<CS>::value> type;
};
template<typename CS>
using vector_t=typename vector_type<CS>::type;
template<typename CS>
using vector_type_t=typename vector_type<CS>::type;
template<typename CS>
struct covector_type
{
	typedef nTuple <Real, dimension<CS>::value> type;
};
template<typename CS>
using covector_t=typename vector_type<CS>::type;
template<typename CS>
using covector_type_t=typename vector_type<CS>::type;

template<typename CS>
struct dimension
{
	static constexpr int value = 3;
};

template<>
struct dimension<geometry::coordinate_system::Polar>
{
	static constexpr int value = 2;
};
template<int N>
struct dimension<geometry::coordinate_system::Cartesian<N>>
{
	static constexpr int value = N;
};

template<typename CS>
struct coordinate_type
{
	typedef Real type;
};
/**
 * if coordinate system is state-less value=true
 *  else value = false
 */
template<typename CS>
struct is_homogeneous
{
	static constexpr bool value = true;
};
template<>
struct is_homogeneous<geometry::coordinate_system::Toroidal>
{
	static constexpr bool value = false;
};

template<>
struct is_homogeneous<geometry::coordinate_system::MagneticFLux>
{
	static constexpr bool value = false;
};
template<int N>
struct typename_as_string<geometry::coordinate_system::Cartesian<N>>
{
	static constexpr char value[] = "Cartesian";
};
template<>
struct typename_as_string<geometry::coordinate_system::Spherical>
{
	static constexpr char value[] = "Spherical";
};

template<int PhiAXIS>
struct typename_as_string<geometry::coordinate_system::Cylindrical<PhiAXIS>>
{
	static constexpr char value[] = "Cylindrical";
};
template<typename TM>
struct ZAxis : public std::integral_constant<int, 2>
{
};
template<int PhiAXIS>
struct ZAxis<geometry::coordinate_system::Cylindrical<PhiAXIS>> : public std::integral_constant<
		int, (PhiAXIS + 2) % 3>
{
};

template<int NDIMS, int ZAXIS>
struct ZAxis<geometry::coordinate_system::Cartesian<NDIMS, ZAXIS>> : public std::integral_constant<
		int, ZAXIS>
{
};
}  // namespace traits
}   // namespace geometry
}   // namespace simpla

#endif /* CORE_GEOMETRY_COORDINATE_SYSTEM_H_ */
