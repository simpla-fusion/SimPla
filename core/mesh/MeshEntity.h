/**
 * @file MeshEntity.h
 * @author salmon
 * @date 2016-05-18.
 */

#ifndef SIMPLA_MESH_MESHENTITY_H
#define SIMPLA_MESH_MESHENTITY_H
namespace simpla { namespace mesh
{
enum MeshEntity
{
    VERTEX = 000, EDGE = 001,

    TRIANGLE = 3,

    QUADRILATERAL = 4,

    // place holder

    POLYGON = 1 << 5,

    // custom polygon

    MAX_POLYGON = 1 << 6,

    TETRAHEDRON = MAX_POLYGON + 6,
    PYRAMID,
    PRISM,
    KNIFE,

    HEXAHEDRON = MAX_POLYGON + 12,
    // place holder
            POLYHEDRON = MAX_POLYGON + (1 << 5),
    // custom POLYHEDRON

    MAX_POLYHEDRON = MAX_POLYGON + (1 << 6)

};

typedef size_t mesh_entity_id_t;
typedef ptrdiff_t mesh_entity_id_diff_t;
}} //namespace simpla { namespace mesh

#endif //SIMPLA_MESH_MESHENTITY_H
