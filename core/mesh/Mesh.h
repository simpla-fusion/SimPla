/**
* @file mesh.h
* @author salmon
* @date 2015-10-16.
*/

#ifndef SIMPLA_MESH_H
#define SIMPLA_MESH_H

#include "../gtl/type_traits.h"
#include "../model/CoordinateSystem.h"

namespace simpla { namespace mesh {
/**
 *  @ingroup diff_geo
 *  @addtogroup  mesh mesh
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
 *  point_type						| DataType of configuration space point (coordinates i.e. (x,y,z)
 *  id_type						    | DataType of grid point's index
 *
 *
 *   @} */

template<typename ...>
class Mesh;

typedef size_t mesh_entity_id_t;

typedef nTuple<size_t, 3> index_type;

typedef nTuple<Real, 3> point_type;

typedef nTuple<Real, 3> vector_type;

typedef std::tuple<point_type, point_type> box_type;

typedef long difference_type;

typedef nTuple<index_type, 3> index_tuple;

typedef std::tuple<index_tuple, index_tuple> index_box_type;

template<typename ...>
struct Mesh
{
    struct Range
    {
        struct iterator;
    };
};

}}//namespace mesh}//namespace simpla

namespace simpla { namespace geometry { namespace traits {

template<typename ...T>
struct metric_type<::simpla::mesh::Mesh<T...>>
{
    typedef ::simpla::traits::unpack_t<0, T...> type;

};
template<typename ...T>
struct coordinate_system_type<::simpla::mesh::Mesh<T...>>
{
    typedef coordinate_system_t<metric_t<::simpla::mesh::Mesh<T...> >> type;

};


}}//namespace geometry{namespace traits{

}//namespace simpla

#endif //SIMPLA_MESH_H
