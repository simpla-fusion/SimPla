/**
 * @file MeshAtlas.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHATLAS_H
#define SIMPLA_MESH_MESHATLAS_H

#include <simpla/algebra/nTuple.h>
#include <simpla/concept/Configurable.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/toolbox/Log.h>
#include <type_traits>
#include "SPObject.h"
#include "MeshBlock.h"

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
class Atlas : public SPObject, public concept::Printable {
   public:
    Atlas();
    virtual ~Atlas();
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const { return os; }

    virtual bool Update();
    size_type GetNumOfLevels() const;

    void SetDx(point_type const &);
    point_type const &GetDx(int level = 0);
    void SetOrigin(point_type const &);
    point_type const &GetOrigin() const;
    void SetBox(box_type const &);
    box_type const &GetBox() const;

    index_box_type FitIndexBox(box_type const &b, int level = 0, int flag = 0) const;

    MeshBlock const &AddBlock(box_type const &, int level = 0);
    MeshBlock const &AddBlock(index_box_type const &, int level = 0);
    MeshBlock const &AddBlock(MeshBlock const &);
    size_type EraseBlock(id_type, int level = 0);
    size_type EraseBlock(MeshBlock const &);
    MeshBlock const &GetBlock(id_type, int level = 0) const;
    MeshBlock const &CoarsenBlock(id_type, int level = 0);
    MeshBlock const &CoarsenBlock(MeshBlock const &);
    MeshBlock const &RefineBlock(id_type, box_type const &, int level = 0);
    MeshBlock const &RefineBlock(MeshBlock const &, box_type const &);
    MeshBlock const &RefineBlock(id_type, index_box_type const &, int level = 0);
    MeshBlock const &RefineBlock(MeshBlock const &, index_box_type const &);

    void Accept(std::function<void(MeshBlock const &)> const &fun, int level = 0) const;
    std::map<id_type, MeshBlock> const &GetBlockList(int level = 0) const;

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

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}  // namespace simpla{namespace mesh_as{

#endif  // SIMPLA_MESH_MESHATLAS_H
