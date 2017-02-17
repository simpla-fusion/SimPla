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
class Atlas : public concept::Printable, public concept::Serializable, public concept::Configurable {
   public:
    Atlas();
    virtual ~Atlas();
    virtual std::ostream &Print(std::ostream &os, int indent) const;
    virtual void Load(const data::DataTable &);
    virtual void Save(data::DataTable *) const;
    size_type count(int level) const;
    void max_level(int);
    int max_level() const;
    bool has(id_type id) const;
    MeshBlock *find(id_type id);
    MeshBlock const *find(id_type id) const;
    MeshBlock *at(id_type id);
    MeshBlock const *at(id_type id) const;
    MeshBlock const *insert(std::shared_ptr<MeshBlock> const &p_m, MeshBlock const *hint = nullptr);
    void link(id_type src, id_type dest){};

    //    std::setValue<id_type> &level(int l);
    //
    //    std::setValue<id_type> const &level(int l) const;
    template <typename TM, typename... Args>
    MeshBlock const *add(Args &&... args) {
        return static_cast<TM const *>(insert(std::make_shared<TM>(std::forward<Args>(args)...), nullptr));
    };

    template <typename... Args>
    MeshBlock const *create(int inc_level, MeshBlock const *hint, Args &&... args) {
        return insert(hint->create(inc_level, std::forward<Args>(args)...), hint);
    };

    template <typename... Args>
    MeshBlock const *create(int inc, id_type h, Args &&... args) {
        create(inc, at(h), std::forward<Args>(args)...);
    };

    template <typename... Args>
    MeshBlock const *clone(Args &&... args) {
        create(0, std::forward<Args>(args)...);
    };

    template <typename... Args>
    MeshBlock const *refine(Args &&... args) {
        create(1, std::forward<Args>(args)...);
    };

    template <typename... Args>
    MeshBlock const *coarsen(Args &&... args) {
        create(-1, std::forward<Args>(args)...);
    };

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}  // namespace simpla{namespace mesh_as{

#endif  // SIMPLA_MESH_MESHATLAS_H
