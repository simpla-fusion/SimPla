/**
 * @file MeshAtlas.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHATLAS_H
#define SIMPLA_MESH_MESHATLAS_H

#include <simpla/algebra/nTuple.h>
#include <simpla/concept/Printable.h>
#include <simpla/geometry/GeoObject.h>
#include <simpla/utilities/Log.h>
#include <type_traits>
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

class Chart;
class MeshBlock;
class Patch;
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

    std::shared_ptr<data::DataTable> Serialize() const override;

    void Deserialize(const std::shared_ptr<data::DataTable> &cfg) override;

    void SetUp() override;

    void Decompose(size_tuple const &d, int local_id = -1);

    index_box_type FitIndexBox(box_type const &b, int level = 0, int flag = 0) const;

    size_type DeletePatch(id_type);
    id_type PushPatch(std::shared_ptr<Patch>);
    std::shared_ptr<Patch> PopPatch(id_type id);

    //    std::shared_ptr<Patch> RefineBlock(id_type, index_box_type const &);
    //    std::set<std::shared_ptr<Patch>> const &Level(int level = 0) const;
    //    void Foreach(std::function<void(std::shared_ptr<MeshBlock>)> const &fun, int level = 0) const;

    void SetPeriodicDimension(size_tuple const &d);
    size_tuple const &GetPeriodicDimension() const;

    size_type GetNumOfLevel() const;

    void SetMaxLevel(size_type l = 1);
    size_type GetMaxLevel() const;

    void SetRefineRatio(size_tuple const &v, size_type level = 0);
    size_tuple GetRefineRatio(int l) const;

    void SetLargestDimensions(size_tuple const &d);
    size_tuple GetLargestDimensions() const;

    void SetSmallestDimensions(size_tuple const &d);
    size_tuple GetSmallestDimensions() const;

    size_tuple GetDimensions() const;

    void SetIndexBox(index_box_type);
    index_box_type GetIndexBox() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
//    virtual void Load(const data::DataTable &);
//    virtual void Save(data::DataTable *) const;
//    size_type size(int level) const;
//    void max_level(int);
//    int max_level() const;
//    bool has(id_type id) const;
//    RectMesh *find(id_type id);
//    RectMesh const *find(id_type id) const;
//    RectMesh *at(id_type id);
//    RectMesh const *at(id_type id) const;
//    RectMesh const *Connect(std::shared_ptr<RectMesh> const &p_m, RectMesh const *hint = nullptr);
//    void link(id_type src, id_type dest){};
//
//    //    std::SetValue<id_type> &level(int l);
//    //
//    //    std::SetValue<id_type> const &level(int l) const;
//    template <typename TM, typename... Args>
//    RectMesh const *add(Args &&... args) {
//        return dynamic_cast<TM const *>(Connect(std::make_shared<TM>(std::forward<Args>(args)...), nullptr));
//    };
//
////    template <typename... Args>
////    MeshBlock const *create(int inc_level, MeshBlock const *hint, Args &&... args) {
////        return insert(hint->create(inc_level, std::forward<Args>(args)...), hint);
////    };
//
//    template <typename... Args>
//    RectMesh const *create(int inc, id_type h, Args &&... args) {
//        create(inc, at(h), std::forward<Args>(args)...);
//    };
//
//    template <typename... Args>
//    RectMesh const *clone(Args &&... args) {
//        create(0, std::forward<Args>(args)...);
//    };
//
//    template <typename... Args>
//    RectMesh const *refine(Args &&... args) {
//        create(1, std::forward<Args>(args)...);
//    };
//
//    template <typename... Args>
//    RectMesh const *coarsen(Args &&... args) {
//        create(-1, std::forward<Args>(args)...);
//    };
}
}  // namespace simpla{namespace mesh_as{

#endif  // SIMPLA_MESH_MESHATLAS_H
