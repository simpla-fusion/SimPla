//
// Created by salmon on 16-6-6.
//

#ifndef SIMPLA_MESHCOMMON_H
#define SIMPLA_MESHCOMMON_H

#include <boost/uuid/uuid.hpp>
#include "../gtl/primitives.h"
#include "../gtl/nTuple.h"

namespace simpla { namespace mesh
{

/**
 *  @ingroup diff_geo
 *  @addtogroup  mesh get_mesh
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



//typedef size_t id_type; //!< Data type  of entity id

typedef nTuple<Real, 3ul> point_type; //!< DataType of configuration space point (coordinates i.e. (x,y,z) )

typedef nTuple<Real, 3ul> vector_type;

typedef std::tuple<point_type, point_type> box_type; //! two corner of rectangle (or hexahedron ) , <lower ,upper>

typedef long index_type; //!< Data type of vertex's index , i.e. i,j

typedef long difference_type; //!< Data type of the difference between indices,i.e.  s = i - j

typedef nTuple<index_type, 3> index_tuple;

typedef std::tuple<index_tuple, index_tuple> index_box_type;

typedef unsigned long MeshEntityId;

typedef long MeshEntityIdDiff;
enum MeshEntityType
{
    VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3

//    TRIANGLE = (3 << 2) | 2,
//
//    QUADRILATERAL = (4 << 2) | 2,
//
//    // place Holder
//
//    POLYGON = ((-1) << 2) | 2,
//
//    // custom polygon
//
//
//
//    TETRAHEDRON = (6 << 2) | 3,
//    PYRAMID,
//    PRISM,
//    KNIFE,
//
//    HEXAHEDRON = MAX_POLYGON + 12,
//    // place Holder
//            POLYHEDRON = MAX_POLYGON + (1 << 5),
//    // custom POLYHEDRON
//
//    MAX_POLYHEDRON = MAX_POLYGON + (1 << 6)

};
enum MeshEntityStatus
{
    INVALID = 0x00, //                          0b000000
    VALID = 0x0F, //                            0b001111 NOT_SHARED| SHARED | OWNED | NOT_OWNED
    OWNED = 0x01, //                            0b000001 owned by local get_mesh block
    NOT_OWNED = 0x02, //                        0b000010 not owned by local get_mesh block
    SHARED = 0x04, //                           0b000100 shared by two or more get_mesh blocks
    NOT_SHARED = 0x08, //                       0b001000 not shared by other get_mesh blocks
    LOCAL = NOT_SHARED | OWNED, //              0b001001
    GHOST = SHARED | NOT_OWNED, //              0b000110
    NON_LOCAL = SHARED | OWNED, //              0b000101
    AFFECTED = GHOST | NON_LOCAL, //            0b000111
    INTERFACE = 0x10, //                        0b010000 interface(boundary) shared by two get_mesh blocks,
    UNDEFINED = 0xFFFF
};


typedef boost::uuids::uuid MeshBlockId;


template<typename ...> struct Mesh;
}}//namespace simpla{namespace get_mesh{
#endif //SIMPLA_MESHCOMMON_H
