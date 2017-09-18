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

#include "EngineObject.h"

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

class MeshBlock;

/**
* @brief
*  - Define topology relation between  mesh blocks
*  - Atlas is defined in a dimensionless topology space , without metric information
*  - '''Origin''' is the origin point of continue topology space and discrete index space
*  - '''dx''' is the resolution ratio  of discrete mesh, x = i * dx + r where 0<= r < dx
*/
class Atlas : public EngineObject {
    SP_OBJECT_HEAD(Atlas, EngineObject)

   public:
    int Foreach(std::function<void(std::shared_ptr<MeshBlock> const &)> const &);

    //    int GetNumOfLevel() const;
    std::shared_ptr<geometry::Chart> GetChart() const;
    void SetChart(std::shared_ptr<geometry::Chart> const &);
    template <typename U, typename... Args>
    void SetChart(Args &&... args) {
        static_assert(std::is_base_of<geometry::Chart, U>::value, "illegal chart type!");
        SetChart(U::New(std::forward<Args>(args)...));
    };
    void SetBoundingBox(box_type const &);
    box_type GetBoundingBox() const;
    index_box_type GetIndexBox(int tag);

    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;

    SP_OBJECT_PROPERTY(int, MaxLevel);
    SP_OBJECT_PROPERTY(index_tuple, RefineRatio);
    SP_OBJECT_PROPERTY(index_tuple, LargestPatchDimensions);
    SP_OBJECT_PROPERTY(index_tuple, SmallestPatchDimensions);
    SP_OBJECT_PROPERTY(index_tuple, PeriodicDimensions);
    SP_OBJECT_PROPERTY(index_tuple, CoarsestIndexBox);

    size_type AddBlock(std::shared_ptr<MeshBlock> const &blk);
    size_type DeleteBlock(id_type);
    std::shared_ptr<MeshBlock> GetBlock(id_type) const;

    void Synchronize(int level, std::map<id_type, std::shared_ptr<data::DataNode>> &) const;
    void Refine(int level, std::map<id_type, std::shared_ptr<data::DataNode>> &) const;
    void Coarsen(int level, std::map<id_type, std::shared_ptr<data::DataNode>> &) const;
};
}  // namespace engine
}  // namespace simpla{namespace mesh_as{

#endif  // SIMPLA_MESH_MESHATLAS_H
