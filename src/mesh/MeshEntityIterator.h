//
// Created by salmon on 16-5-24.
//

#ifndef SIMPLA_MESHENTITYITERATOR_H
#define SIMPLA_MESHENTITYITERATOR_H

#include "Mesh.h"
#include "../gtl/iterator/RandomAccessIterator.h"
#include "../gtl/iterator/IteratorAdapter.h"
#include "../gtl/iterator/Range.h"

namespace simpla { namespace mesh
{
typedef IteratorAdapter<RandomAccessIterator<MeshEntityId, MeshEntityIdDiff>> MeshEntityIterator;

typedef Range<MeshEntityIterator> MeshEntityRange;

}}//namespace simpla { namespace mesh {

#endif //SIMPLA_MESHENTITYITERATOR_H
