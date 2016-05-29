/**
 * @file MeshEntity.h
 * @author salmon
 * @date 2016-05-18.
 */

#ifndef SIMPLA_MESH_MESHENTITY_H
#define SIMPLA_MESH_MESHENTITY_H

#include <cassert>
#include "Mesh.h"
#include "../gtl/iterator/RandomAccessIterator.h"
#include "../gtl/iterator/IteratorAdapter.h"
#include "../gtl/iterator/RangeAdapter.h"


namespace simpla { namespace mesh
{
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


typedef IteratorAdapter<std::random_access_iterator_tag, MeshEntityId, MeshEntityIdDiff, MeshEntityId *, MeshEntityId> MeshEntityIterator;


typedef RangeAdapter<MeshEntityIterator> MeshEntityRange;

}} //namespace simpla { namespace mesh

#endif //SIMPLA_MESH_MESHENTITY_H
