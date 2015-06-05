/**
 * @file mesh.h
 *
 * @date 2015年2月9日
 * @author salmon
 */

#ifndef CORE_MESH_MESH_H_
#define CORE_MESH_MESH_H_
#include "structured.h"
#include "structured/fdm.h"
#include "structured/interpolator.h"
#include "domain.h"

#include "../geometry/coordinate_system.h"
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
 *  coordinate_tuple				| datatype of coordinates
 *  id_type						    | datatype of grid point's index
 *
 *
 * @}

 */

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
template<size_t NDIMS>
using CartesianRectMesh=StructuredMesh<geometry::coordinate_system::Cartesian<NDIMS>,
InterpolatorLinear, FiniteDiffMethod>;

}  // namespace simpla

#endif /* CORE_MESH_MESH_H_ */
