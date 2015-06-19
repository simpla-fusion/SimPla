/**
 * @file mesh.h
 *
 * @date 2015年2月9日
 * @author salmon
 */

#ifndef CORE_MESH_MESH_H_
#define CORE_MESH_MESH_H_

#include <iostream>
#include <memory>

#include "../geometry/coordinate_system.h"
#include "../gtl/type_traits.h"
#include "mesh_traits.h"

namespace simpla
{
/**
 * @ingroup diff_geo
 * @{
 */

/**
 *  @ingroup diff_geo
 *  @addtogroup  topology Topology
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

template<typename ...> struct Mesh;

template<typename ... T>
std::ostream & operator<<(std::ostream & os, Mesh<T...> const &d)
{
	return d.print(os);
}
//template<typename TM, typename ... Args>
//std::shared_ptr<TM> make_mesh(Args && ...args)
//{
//	return std::make_shared<TM>(std::forward<Args>(args)...);
//}
template<typename TM>
std::shared_ptr<TM> make_mesh()
{
	return std::make_shared<TM>();
}

namespace traits
{

template<typename CS, typename TAG>
struct type_id<Mesh<CS, TAG> >
{
};
template<typename CS, typename ... T>
struct coordinate_system_type<Mesh<CS, T...>>
{
	typedef CS type;
};

template<typename ...T>
struct rank<Mesh<T...> > : public rank<
		typename coordinate_system_type<Mesh<T...> >::type>::type
{
};

template<typename ...Others>
struct ZAxis<Mesh<Others...> > : public geometry::traits::ZAxis<
		coordinate_system_type<typename Mesh<Others...>::type> >
{
};
}  // namespace traits

}  // namespace simpla

#endif /* CORE_MESH_MESH_H_ */
