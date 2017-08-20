/**
 * @file MeshAtlas.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHATLAS_H
#define SIMPLA_MESH_MESHATLAS_H

#include "simpla/SIMPLA_config.h"

#include <type_traits>

#include "simpla/algebra/nTuple.ext.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/geometry/Chart.h"
#include "simpla/utilities/Log.h"

#include "Patch.h"
#include "SPObject.h"

namespace simpla {
namespace engine {

/**
 *  Manifold (Differential Manifold):
 *  A presentation of a _topological manifold_ is a second countable Hausdorff space that is locally homeomorphic
 *  to a linear space, by a collection (called an atlas) of homeomorphisms called _charts_. The composition of one
 *  _chart_ with the inverse of another chart is a function called a _transition map_, and defines a homeomorphism
 *  of an open subset of the linear space onto another open subset of the linear space.
 */

/**
 * @startuml
 * participant A
 * loop until t >= stop_time
 *      alt  refine condition = YES
 *           create B
 *           A -> B : create
 *           create C
 *           A -> C : create
 *      end
 *      A -> B : sync
 *      A -> C : sync
 *      A -> A: t += dt
 *      loop  m times
 *           B -> B: t += dt/m
 *           C-> C:  t += dt/m
 *           B-> C: sync
 *           C-> B: sync
 *      end
 *      B -> A : sync
 *      C -> A : sync
 *      alt  coarsen condition == YES
 *          C --> A : Coarsen
 *          destroy C
 *      end
 * end
 *
 * @enduml
 */

class Patch;
class MeshBlock;

/**
* @brief
*  - Define topology relation between  mesh blocks
*  - Atlas is defined in a dimensionless topology space , without metric information
*  - '''Origin''' is the origin point of continue topology space and discrete index space
*  - '''dx''' is the resolution ratio  of discrete mesh, x = i * dx + r where 0<= r < dx
*/
class Atlas : public SPObject {
    SP_OBJECT_HEAD(Atlas, SPObject)

   public:
    size_type DeletePatch(id_type);
    id_type SetPatch(const std::shared_ptr<Patch> &p);
    std::shared_ptr<Patch> GetPatch(id_type id);
    std::shared_ptr<Patch> GetPatch(const std::shared_ptr<MeshBlock> &mblk);
    std::shared_ptr<const Patch> GetPatch(id_type id) const;

    int GetNumOfLevel() const;

    SP_OBJECT_PROPERTY(int, MaxLevel);
    SP_OBJECT_PROPERTY(index_tuple, RefineRatio);
    SP_OBJECT_PROPERTY(index_tuple, LargestPatchDimensions);
    SP_OBJECT_PROPERTY(index_tuple, SmallestPatchDimensions);
    SP_OBJECT_PROPERTY(index_tuple, PeriodicDimensions);
    SP_OBJECT_PROPERTY(index_tuple, CoarsestIndexBox);
};
}
}  // namespace simpla{namespace mesh_as{

#endif  // SIMPLA_MESH_MESHATLAS_H
