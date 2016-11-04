/**
 * @file MeshAtlas.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHATLAS_H
#define SIMPLA_MESH_MESHATLAS_H

#include <type_traits>
#include "simpla/toolbox/Log.h"
#include "simpla/toolbox/nTuple.h"
#include "MeshCommon.h"
#include "MeshBlock.h"
#include "TransitionMap.h"
#include "Attribute.h"

namespace simpla { namespace mesh
{
class Attribute;

/**
 *  Manifold (Differential Manifold):
 *  A presentation of a _topological manifold_ is a second countable Hausdorff space that is locally homeomorphic
 *  to a linear space, by a collection (called an atlas) of homeomorphisms called _charts_. The composition of one
 *  _chart_ with the inverse of another chart is a function called a _transition map_, and defines a homeomorphism
 *  of an open subset of the linear space onto another open subset of the linear space.
 */


class Atlas
{


public:

    Atlas();

    ~Atlas();

    unsigned int max_level() const;

    bool has(id_type id) const;


    id_type insert(std::shared_ptr<MeshBlock> const p_m, id_type hint = 0);

    id_type insert(MeshBlock &p_m) { return insert(p_m.shared_from_this()); };

    MeshBlock &at(id_type id = 0);

    MeshBlock const &at(id_type id = 0) const;

    MeshBlock &operator[](id_type id) { return at(id); };

    MeshBlock const &operator[](id_type id) const { return at(id); };

    template<typename TM> TM &as(id_type id = 0) { return static_cast<TM &>(at(id)); };

    template<typename TM> TM const &as(id_type id = 0) const { return static_cast<TM const &>(at(id)); };

    template<typename ...Args>
    id_type create(id_type hint, Args &&...args) { return insert(at(hint).create(std::forward<Args>(args)...), hint); };

    void link(id_type src, id_type dest) {};

//    std::set<id_type> &level(int l);
//
//    std::set<id_type> const &level(int l) const;


private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};

}}//namespace simpla{namespace mesh_as{

#endif //SIMPLA_MESH_MESHATLAS_H
