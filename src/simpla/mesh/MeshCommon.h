//
// Created by salmon on 16-6-6.
//

#ifndef SIMPLA_MESHCOMMON_H
#define SIMPLA_MESHCOMMON_H

#include <boost/uuid/uuid.hpp>
#include <simpla/algebra/nTuple.h>
#include <simpla/toolbox/sp_def.h>

namespace simpla { namespace mesh
{

//typedef union { struct { u_int8_t w, z, y, x; }; int32_t v; } MeshEntityId32;

typedef union { struct { u_int16_t w, z, y, x; }; int64_t v; } MeshEntityId;



//typedef MeshEntityId64 MeshEntityId;

#define MAX_POLYGON 20

enum MeshEntityType
{
//    VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3,
//    FIBER = 10 // points in cell

    TRIANGLE = (3 << 2) | 2,

    QUADRILATERAL = (4 << 2) | 2,

    // place RangeHolder

    POLYGON = ((-1) << 2) | 2,

    // custom polygon



    TETRAHEDRON = (6 << 2) | 3,
    PYRAMID,
    PRISM,
    KNIFE,

    HEXAHEDRON = MAX_POLYGON + 12,
    // place RangeHolder
            POLYHEDRON = MAX_POLYGON + (1 << 5),
    // custom POLYHEDRON

    MAX_POLYHEDRON = MAX_POLYGON + (1 << 6)

};
/**
 *   |<-----------------------------     valid   --------------------------------->|
 *   |<- not owned  ->|<-------------------       owned     ---------------------->|
 *   |----------------*----------------*---*---------------------------------------|
 *   |<---- ghost --->|                |   |                                       |
 *   |<------------ shared  ---------->|<--+--------  not shared  ---------------->|
 *   |<------------- DMZ    -------------->|<----------   not DMZ   -------------->|
 *
 */


enum MeshZoneTag
{
    SP_ES_NULL = 0x00, //                          0b000000
    SP_ES_ALL = 0x0F, //                            0b001111 SP_ES_NOT_SHARED| SP_ES_SHARED | SP_ES_OWNED | SP_ES_NOT_OWNED
    SP_ES_OWNED = 0x01, //                            0b000001 owned by local get_mesh block
    SP_ES_NOT_OWNED = 0x02, //                        0b000010 not owned by local get_mesh block
    SP_ES_SHARED = 0x04, //                           0b000100 shared by two or more get_mesh grid_dims
    SP_ES_NOT_SHARED = 0x08, //                       0b001000 not shared by other get_mesh grid_dims
    SP_ES_LOCAL = SP_ES_NOT_SHARED | SP_ES_OWNED, //              0b001001
    SP_ES_GHOST = SP_ES_SHARED | SP_ES_NOT_OWNED, //              0b000110
    SP_ES_NON_LOCAL = SP_ES_SHARED | SP_ES_OWNED, //              0b000101
    SP_ES_INTERFACE = 0x010, //                        0b010000 interface(boundary) shared by two get_mesh grid_dims,
    SP_ES_DMZ = 0x100,
    SP_ES_NOT_DMZ = 0x200,
    SP_ES_VALID = 0x400,
    SP_ES_UNDEFINED = 0xFFFF
};



/**
 *  @ingroup diff_geo
 *  @addtogroup  mesh get_mesh
 *  @{
 *  Mesh<>
 *  Concept:
 *  - Mesh<> know local information of topology and vertex coordinates, and
 *  - only explicitly store vertex adjacencies;
 *  - Mesh<> do not know global coordinates, topology_dims;
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







}}//namespace simpla{namespace get_mesh{
#endif //SIMPLA_MESHCOMMON_H
