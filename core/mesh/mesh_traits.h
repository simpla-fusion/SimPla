/**
 * @file mesh_traits.h
 *
 * @date 2015年6月19日
 * @author salmon
 */

#ifndef CORE_MESH_MESH_TRAITS_H_
#define CORE_MESH_MESH_TRAITS_H_

#include <cstdbool>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "../gtl/type_traits.h"

namespace simpla
{




/**
 *  @ingroup diff_geo
 *  @addtogroup  Topology Topology
 *  @{
 *  @brief   connection between discrete points
 *
 *  ## Summary
 *
 *
 * Cell shapes supported in '''libmesh''' http://libmesh.github.io/doxygen/index.html
 * - 3 and 6 nodes triangles (Tri3, Tri6)
 * - 4, 8, and 9 nodes quadrilaterals (Quad4, Quad8, Quad9)
 * - 4 and 6 nodes infinite quadrilaterals (InfQuad4, InfQuad6)
 * - 4 and 10 nodes tetrahedrals (Tet4, Tet10)
 * - 8, 20, and 27 nodes  hexahedrals (Hex8, Hex20, Hex27)
 * - 6, 15, and 18 nodes prisms (Prism6, Prism15, Prism18)
 * - 5 nodes  pyramids (Pyramid5)
 * - 8, 16, and 18 nodes  infinite hexahedrals (InfHex8, InfHex16, InfHex18) ??
 * - 6 and 12 nodes  infinite prisms (InfPrism6, InfPrism12) ??
 *
 *
 * ## Note
 * - the width of unit cell is 1;
 *
 *
 *  Member type	 				    | Semantics
 *  --------------------------------|--------------
 *  point_type						| datatype of configuration space point (coordinates i.e. (x,y,z)
 *  id_type						    | datatype of grid point's index
 *
 *
 * @}

 */
template<typename ...> struct Topology;

/**
 *
 */
template<typename ...> struct Mesh;

template<typename ... T>
std::ostream & operator<<(std::ostream & os, Mesh<T...> const &d)
{
	return d.print(os);
}

template<typename ...T>
std::shared_ptr<Mesh<T...>> make_mesh()
{
	return std::make_shared<Mesh<T...>>();
}
/**
 *  Default value of Mesh are defined following
 */
namespace traits
{
template<typename > struct type_id;

template<typename > struct is_mesh;
template<typename > struct mesh_type;
template<typename T> using mesh_type_t= typename mesh_type<T>::type;

template<typename > struct id_type;
template<typename T> using id_type_t= typename id_type<T>::type;

template<typename > struct coordinate_system_type;
template<typename T> using coordinate_system_t= typename coordinate_system_type<T >::type;

template<typename > struct scalar_type;
template<typename T> using scalar_type_t= typename scalar_type<T >::type;

template<typename > struct point_type;
template<typename T> using point_type_t= typename point_type<T>::type;

template<typename > struct vector_type;
template<typename T> using vector_type_t= typename vector_type<T>::type;

template<typename > struct rank;
template<typename > struct ZAxis;

template<typename ... T>
struct type_id<Mesh<T...> >
{
	static const std::string name()
	{
		return "Mesh<+" + type_id<coordinate_system_t<Mesh<T...> > >::name()
				+ " >";
	}
};

template<typename T>
struct is_mesh: public std::integral_constant<bool, false>
{
};
template<typename ...T>
struct is_mesh<Mesh<T...>> : public std::integral_constant<bool, true>
{
};

template<typename T>
struct mesh_type
{
	typedef std::nullptr_t type;
};

template<typename T>
struct id_type
{
	typedef int64_t type;
};

template<typename ...T>
struct id_type<Mesh<T...> >
{
	typedef std::uint64_t type;
};

template<typename T>
struct coordinate_system_type
{
	typedef std::nullptr_t type;
};
template<typename CS, typename ... T>
struct coordinate_system_type<Mesh<CS, T...>>
{
	typedef CS type;
};

template<typename ...T>
struct scalar_type<Mesh<T...> >
{
	typedef typename Mesh<T...>::scalar_type type;
};
template<typename ...T>
struct point_type<Mesh<T...> >
{
	typedef typename Mesh<T...>::point_type type;
};

template<typename ...T>
struct vector_type<Mesh<T...> >
{
	typedef typename Mesh<T...>::vector_type type;
};

template<typename ...T>
struct rank<Mesh<T...> > : public std::integral_constant<size_t,
		Mesh<T...>::ndims>
{
};

template<typename ...T>
struct ZAxis<Mesh<T...> > : public std::integral_constant<size_t,
		Mesh<T...>::ZAXIS>
{
};

}  // namespace traits

}  // namespace simpla

#endif /* CORE_MESH_MESH_TRAITS_H_ */
