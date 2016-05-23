/**
* @file mesh.h
* @author salmon
* @date 2015-10-16.
*/

#ifndef SIMPLA_MESH_H
#define SIMPLA_MESH_H


#include "../gtl/ntuple.h"
#include "../gtl/type_traits.h"
#include "../gtl/primitives.h"


namespace simpla { namespace mesh
{
/**
 *  @ingroup diff_geo
 *  @addtogroup  mesh mesh
 *  @{
 *  Mesh<>
 *  Concept:
 *  - Mesh<> know local information of topology and vertex coordinates, and
 *  - only explicitly store vertex adjacencies;
 *  - Mesh<> do not know global coordinates, topology;
 *  - Mesh<> do not know metric;
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
 *
 *
 *   @} */



enum EntityType
{
    VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3

//    TRIANGLE = (3 << 2) | 2,
//
//    QUADRILATERAL = (4 << 2) | 2,
//
//    // place holder
//
//    POLYGON = ((-1) << 2) | 2,
//
//    // custom polygon
//
//    TETRAHEDRON = (6 << 2) | 3,
//    PYRAMID,
//    PRISM,
//    KNIFE,
//
//    HEXAHEDRON = MAX_POLYGON + 12,
//    // place holder
//            POLYHEDRON = MAX_POLYGON + (1 << 5),
//    // custom POLYHEDRON
//
//    MAX_POLYHEDRON = MAX_POLYGON + (1 << 6)

};

typedef size_t id_type; //!< Data type  of entity id

typedef gtl::nTuple<Real, 3> point_type; //!< DataType of configuration space point (coordinates i.e. (x,y,z) )

typedef gtl::nTuple<Real, 3> vector_type;

typedef std::tuple<point_type, point_type> box_type; //! two corner of rectangle (or hexahedron ) , <lower ,upper>

typedef long index_type; //!< Data type of vertex's index , i.e. i,j

typedef long difference_type; //!< Data type of the difference between indices,i.e.  s = i - j

typedef gtl::nTuple<index_type, 3> index_tuple;

typedef std::tuple<index_tuple, index_tuple> index_box_type;

class EntityIterator;

class EntityRange;

typedef unsigned long EntityId;

typedef unsigned long mesh_id;

class MeshBase;

class MeshUpdater;

class MeshAtlas;

class MeshAttribute;

class MeshConnection;

template<typename ...> struct Mesh;


}}//namespace mesh}//namespace simpla



#endif //SIMPLA_MESH_H
