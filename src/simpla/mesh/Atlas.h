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
class AttributeBase;

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

    unsigned int level() const;

    bool has(id_type id) const;


    virtual id_type insert(std::shared_ptr<MeshBlock> const p_m);

    id_type insert(MeshBlock &p_m) { return insert(p_m.shared_from_this()); };

    virtual MeshBlock &at(id_type id = 0);

    virtual MeshBlock const &at(id_type id = 0) const;

    MeshBlock &operator[](id_type id) { return at(id); };

    MeshBlock const &operator[](id_type id) const { return at(id); };

    template<typename TM> TM &as(id_type id = 0) { return static_cast<TM &>(at(id)); };

    template<typename TM> TM const &as(id_type id = 0) const { return static_cast<TM const &>(at(id)); };

    /**
     *  if '''has(hint)''' then '''at(hint).create(level,b)'''
     *  else find_overlap(b,level)->create(level,b)
     */
    virtual id_type create(int level, index_box_type const &b, id_type hint = 0);

    void update(id_type id);


    virtual void deploy(id_type id = 0);

    virtual void erase(id_type id);

    virtual void clear(id_type id);

    virtual void coarsen(id_type dest, id_type src);

    virtual void update(id_type dest, id_type src);

    /**
     * @brief
     * @param i0
     * @param i1
     * @return  -1 => refine
     *           0 => adjointing
     *           1 => coarsen
     */
    int link(id_type i0, id_type i1);


    std::set<id_type> &level(int l) { return m_layer_[l]; }

    std::set<id_type> const &level(int l) const { return m_layer_[l]; }

    multi_links_type same_level(id_type id) const { return m_adjacent_.equal_range(id); };

    multi_links_type upper_level(id_type id) const { return m_refine_.equal_range(id); };

    multi_links_type lower_lelvel(id_type id) const { return m_coarsen_.equal_range(id); };

    void update_all();

    void register_attribute(std::shared_ptr<AttributeBase> attr);

    void unregister_attribute(std::shared_ptr<AttributeBase> attr);


private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};
}}//namespace simpla{namespace mesh_as{

#endif //SIMPLA_MESH_MESHATLAS_H
