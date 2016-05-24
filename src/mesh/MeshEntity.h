/**
 * @file MeshEntity.h
 * @author salmon
 * @date 2016-05-18.
 */

#ifndef SIMPLA_MESH_MESHENTITY_H
#define SIMPLA_MESH_MESHENTITY_H
namespace simpla { namespace mesh
{
enum MeshEntityType
{
    VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 4

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
//
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

typedef unsigned long MeshEntityId;
typedef long MeshEntityIdDiff;
}} //namespace simpla { namespace mesh

#endif //SIMPLA_MESH_MESHENTITY_H
