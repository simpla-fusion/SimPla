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

namespace simpla { namespace mesh
{

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
    typedef typename MeshBlock::id_type id_type;
private:
    static constexpr int MAX_NUM_OF_LEVEL = 10;

    typedef typename std::multimap<id_type, id_type>::iterator link_iterator;
    typedef typename std::multimap<id_type, id_type>::const_iterator const_link_iterator;
    typedef std::pair<const_link_iterator, const_link_iterator> multi_links_type;
    std::map<id_type, std::shared_ptr<MeshBlock>> m_nodes_;
    std::multimap<id_type, id_type> m_adjacent_;
    std::multimap<id_type, id_type> m_refine_;
    std::multimap<id_type, id_type> m_coarsen_;
    std::set<id_type> m_layer_[MAX_NUM_OF_LEVEL];
    unsigned int m_max_level_ = 0;
public:

    Atlas();

    ~Atlas();

    unsigned int max_level() const { return m_max_level_; }


    std::shared_ptr<MeshBlock> at(id_type id) { try { return m_nodes_.at(id); } catch (...) { return nullptr; }};

    std::shared_ptr<const MeshBlock> at(id_type id) const
    {
        try { return m_nodes_.at(id); } catch (...) { return nullptr; }
    };

    std::shared_ptr<MeshBlock> operator[](id_type id) { return at(id); };

    std::shared_ptr<const MeshBlock> operator[](id_type id) const { return at(id); };

    MeshBlock const *mesh(id_type id = 0) const
    {
        MeshBlock const *res = nullptr;

        if (!m_nodes_.empty()) { if (id == 0) { res = m_nodes_.begin()->second.get(); } else { res = &(*at(id)); }}

        return res;

    };

    bool has(id_type id) const { return m_nodes_.find(id) != m_nodes_.end(); }

    void add(std::shared_ptr<MeshBlock> const p_m);

    void update(id_type id);

    /**
     * @brief
     * @param i0
     * @param i1
     * @return  -1 => refine
     *           0 => adjointing
     *           1 => coarsen
     */
    int link(id_type i0, id_type i1);

    void unlink(id_type id);

    void erase(id_type m_id);

    std::set<id_type> &level(int l) { return m_layer_[l]; }

    std::set<id_type> const &level(int l) const { return m_layer_[l]; }

    multi_links_type same_level(id_type id) const { return m_adjacent_.equal_range(id); };

    multi_links_type upper_level(id_type id) const { return m_refine_.equal_range(id); };

    multi_links_type lower_lelvel(id_type id) const { return m_coarsen_.equal_range(id); };

    void update_all();

};
}}//namespace simpla{namespace mesh_as{

#endif //SIMPLA_MESH_MESHATLAS_H
