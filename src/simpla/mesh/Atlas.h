/**
 * @file MeshAtlas.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHATLAS_H
#define SIMPLA_MESH_MESHATLAS_H

#include <type_traits>
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/nTuple.h>
#include <simpla/concept/Printable.h>
#include "MeshCommon.h"
#include "MeshBlock.h"

namespace simpla { namespace mesh
{

/**
 *  Manifold (Differential Manifold):
 *  A presentation of a _topological manifold_ is a second countable Hausdorff space that is locally homeomorphic
 *  to a linear space, by a collection (called an atlas) of homeomorphisms called _charts_. The composition of one
 *  _chart_ with the inverse of another chart is a function called a _transition map_, and defines a homeomorphism
 *  of an open subset of the linear space onto another open subset of the linear space.
 */


class Atlas : public concept::Printable
{


public:

    Atlas();

    virtual ~Atlas();

    std::string name() const;

    std::ostream &print(std::ostream &os, int indent) const;


    size_type count(int level) const;

    void max_level(int);

    int max_level() const;

    bool has(id_type id) const;

    MeshBlock *at(id_type id);

    MeshBlock const *at(id_type id) const;

    MeshBlock const *insert(std::shared_ptr<MeshBlock> const &p_m, MeshBlock const *hint = nullptr);

    void link(id_type src, id_type dest) {};

//    std::set<id_type> &level(int l);
//
//    std::set<id_type> const &level(int l) const;
    template<typename TM, typename ... Args>
    MeshBlock const *add(Args &&...args)
    {
        return static_cast<TM const *>(insert(std::make_shared<TM>(std::forward<Args>(args)...), nullptr));
    };

    template<typename ... Args>
    MeshBlock const *create(int inc_level, MeshBlock const *hint, Args &&...args)
    {
        return insert(hint->create(inc_level, std::forward<Args>(args)...), hint);
    };

    template<typename ... Args>
    MeshBlock const *create(int inc, id_type h, Args &&...args) { create(inc, at(h), std::forward<Args>(args)...); };

    template<typename ... Args>
    MeshBlock const *clone(Args &&...args) { create(0, std::forward<Args>(args)...); };

    template<typename ... Args>
    MeshBlock const *refine(Args &&...args) { create(1, std::forward<Args>(args)...); };

    template<typename ... Args>
    MeshBlock const *coarsen(Args &&...args) { create(-1, std::forward<Args>(args)...); };


private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};

}}//namespace simpla{namespace mesh_as{

#endif //SIMPLA_MESH_MESHATLAS_H
