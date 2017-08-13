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
#include "simpla/data/Serializable.h"
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
class Atlas : public SPObject, public data::Serializable {
    SP_OBJECT_HEAD(Atlas, SPObject)
   public:
    Atlas();
    ~Atlas() override;
    SP_DEFAULT_CONSTRUCT(Atlas);

    void Serialize(data::DataTable &cfg) const override;
    void Deserialize(const data::DataTable &cfg) override;

    void DoUpdate() override;

    size_type DeletePatch(id_type);
    id_type SetPatch(Patch &&);
    Patch *GetPatch(id_type id);
    Patch *GetPatch(MeshBlock const &);
    Patch const *GetPatch(id_type id) const;

    int GetNumOfLevel() const;

    void SetMaxLevel(int l = 1);
    int GetMaxLevel() const;

    void SetRefineRatio(nTuple<int, 3> const &v, int level = 0);
    nTuple<int, 3> GetRefineRatio(int l) const;

    void SetLargestPatchDimensions(nTuple<int, 3> const &d);
    nTuple<int, 3> GetLargestPatchDimensions() const;

    void SetSmallestPatchDimensions(nTuple<int, 3> const &d);
    nTuple<int, 3> GetSmallestPatchDimensions() const;

    void SetPeriodicDimensions(nTuple<int, 3> const &t);
    nTuple<int, 3> const &GetPeriodicDimensions() const;

    void SetCoarsestIndexBox(index_box_type const &t);
    index_box_type const &GetCoarsestIndexBox() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}  // namespace simpla{namespace mesh_as{

#endif  // SIMPLA_MESH_MESHATLAS_H
