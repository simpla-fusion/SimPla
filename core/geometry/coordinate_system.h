/**
 * @file coordinate_system.h
 *
 * @date 2015年6月3日
 * @author salmon
 */

#ifndef CORE_GEOMETRY_COORDINATE_SYSTEM_H_
#define CORE_GEOMETRY_COORDINATE_SYSTEM_H_

namespace simpla
{
namespace geometry
{
template<typename, typename > struct map;
template<typename > struct mertic;

namespace traits
{

template<typename > struct dimension;
template<typename > struct coordinate_type;
template<typename > struct is_homogeneous;
template<typename > struct typename_as_string;
template<typename > struct ZAxis;
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
template<size_t IPhiAxis = 2>
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
using scalar_t =typename scalar_type<CS>::type;

template<typename CS>
struct point_type
{
	typedef nTuple<Real, dimension<CS>::value> type;
};
template<typename CS>
using point_t=typename point_type<CS>::type;

template<typename CS>
struct vector_type
{
	typedef nTuple<Real, dimension<CS>::value> type;
};
template<typename CS>
using vector_t=typename vector_type<CS>::type;

template<typename CS>
struct covector_type
{
	typedef nTuple<Real, dimension<CS>::value> type;
};
template<typename CS>
using covector_t=typename vector_type<CS>::type;

template<typename CS>
struct dimension
{
	static constexpr size_t value = 3;
};

template<>
struct dimension<geometry::coordinate_system::Polar>
{
	static constexpr size_t value = 2;
};
template<size_t N>
struct dimension<geometry::coordinate_system::Cartesian<N>>
{
	static constexpr size_t value = N;
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
template<size_t N>
struct typename_as_string<geometry::coordinate_system::Cartesian<N>>
{
	static constexpr char value[] = "Cartesian";
};
template<>
struct typename_as_string<geometry::coordinate_system::Spherical>
{
	static constexpr char value[] = "Spherical";
};

template<size_t PhiAXIS>
struct typename_as_string<geometry::coordinate_system::Cylindrical<PhiAXIS>>
{
	static constexpr char value[] = "Cylindrical";
};
template<typename TM>
struct ZAxis: public std::integral_constant<size_t, 2>
{
};
template<size_t PhiAXIS>
struct ZAxis<geometry::coordinate_system::Cylindrical<PhiAXIS>> : public std::integral_constant<
		size_t, (PhiAXIS + 2) % 3>
{
};

template<size_t NDIMS, size_t ZAXIS>
struct ZAxis<geometry::coordinate_system::Cartesian<NDIMS, ZAXIS>> : public std::integral_constant<
		size_t, ZAXIS>
{
};
}  // namespace traits
}   // namespace geometry
}   // namespace simpla

#endif /* CORE_GEOMETRY_COORDINATE_SYSTEM_H_ */
