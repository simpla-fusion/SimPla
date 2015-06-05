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

namespace traits
{

template<typename > struct dimension;
template<typename > struct coordinate_type;
template<typename > struct is_homogeneous;
template<typename > struct typename_as_string;

}  // namespace traits

namespace coordinate_system
{

struct Polar
{
};

template<size_t N>
struct Cartesian
{
};

struct Spherical
{
};
struct Cylindical
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
struct dimension
{
	static constexpr size_t value = 3;
};

template<>
struct dimension<coordinate_system::Polar>
{
	static constexpr size_t value = 2;
};
template<size_t N>
struct dimension<coordinate_system::Cartesian<N>>
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
struct is_homogeneous<coordinate_system::Toroidal>
{
	static constexpr bool value = false;
};

template<>
struct is_homogeneous<coordinate_system::MagneticFLux>
{
	static constexpr bool value = false;
};
template<size_t N>
struct typename_as_string<coordinate_system::Cartesian<N>>
{
	static constexpr char value[] = "Cartesian";
};
template<>
struct typename_as_string<coordinate_system::Spherical>
{
	static constexpr char value[] = "Spherical";
};

template<>
struct typename_as_string<coordinate_system::Cylindical>
{
	static constexpr char value[] = "Cylindical";
};

}  // namespace traits

}   // namespace geometry
}   // namespace simpla

#endif /* CORE_GEOMETRY_COORDINATE_SYSTEM_H_ */
