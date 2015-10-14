/**
 * @file mesh_traits.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_GEOMETRY_TRAITS_H
#define SIMPLA_GEOMETRY_TRAITS_H

#include <type_traits>
#include <string>

namespace simpla
{


template<typename ...> class Geometry;

namespace traits
{
template<typename> struct type_id;

template<typename> struct is_geometry;
template<typename> struct geometry_type;

template<typename ... T>
struct type_id<Geometry<T...> >
{
	static const std::string name()
	{
		return "Geometry";

//		return "Geometry<+" + type_id<coordinate_system_t < Geometry<T...> > > ::name()
//				+ " >";
	}
};

template<typename T> struct is_geometry : public std::integral_constant<bool, false> { };

template<typename ...T> struct is_geometry<Geometry<T...>> : public std::integral_constant<bool, true>
{
};

template<typename T> struct geometry_type { typedef std::nullptr_t type; };

template<typename ...T> struct id_type<Geometry<T...> >
{
	typedef std::uint64_t type;
};

template<typename CS, typename ... T>
struct coordinate_system_type<Geometry<CS, T...>>
{
	typedef CS type;
};

template<typename ...T>
struct scalar_type<Geometry<T...> >
{
	typedef typename Geometry<T...>::scalar_type type;
};

template<typename ...T>
struct point_type<Geometry<T...> >
{
	typedef typename Geometry<T...>::point_type type;
};

template<typename ...T>
struct vector_type<Geometry<T...> >
{
	typedef typename Geometry<T...>::vector_type type;
};

template<typename ...T>
struct rank<Geometry<T...> > : public std::integral_constant<size_t, Geometry<T...>::ndims>
{
};

template<typename ...T>
struct ZAxis<Geometry<T...> > : public std::integral_constant<size_t, Geometry<T...>::ZAXIS>
{
};

}  // namespace traits

} //namespace simpla

#endif //SIMPLA_GEOMETRY_TRAITS_H
