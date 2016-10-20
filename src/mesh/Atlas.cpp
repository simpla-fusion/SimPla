/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "Atlas.h"
#include "TransitionMap.h"

#include "../toolbox/BoxUtility.h"

namespace simpla { namespace mesh
{
Atlas::Atlas() {};

Atlas::~Atlas() {};

void Atlas::add(std::shared_ptr<DomainBase> const p_m)
{
    m_nodes_.emplace(p_m->id(), p_m);
    update(p_m->id());
};

void Atlas::update(uuid id)
{
    unlink(id);

    auto it = m_nodes_.find(id);

    if (it == m_nodes_.end())
    {
        return;
    } else if (id != it->second->id())
    {
        m_nodes_.erase(it);
        return;
    }
    MeshBase const &m = *(it->second->mesh());
    assert(m.level() < MAX_NUM_OF_LEVEL);
    m_max_level_ = std::max(m_max_level_, m.level());
    m_layer_[m.level()].insert(id);

    //TODO  check overlap and add links
    for (auto const &item:m_nodes_) { if (item.first != id) { link(id, item.first); }}
}

void Atlas::erase(uuid m_id)
{
    unlink(m_id);
    m_nodes_.erase(m_id);
}

int Atlas::link(uuid i0, uuid i1)
{
    assert(has(i0) && has(i1));
    MeshBase const &m0 = mesh(i0);
    MeshBase const &m1 = mesh(i1);
    int l0 = m0.level();
    int l1 = m1.level();
    box_type b0 = m0.box();
    box_type b1 = m1.box();
    vector_type dx = m0.dx();
    vector_type L = m0.period_length();
    switch (l0 - l1)
    {
        case 0:
            if (toolbox::check_adjoining(b0, b1, dx))
            {
                m_adjacent_.emplace(i0, i1);
                m_adjacent_.emplace(i1, i0);
            }
            break;

        case -1:
            if (toolbox::check_overlapping(b0, b1))
            {
                m_refine_.emplace(i0, i1);
                m_coarsen_.emplace(i1, i0);
            }
            break;
        case 1:
            if (toolbox::check_overlapping(b0, b1))
            {
                m_coarsen_.emplace(i0, i1);
                m_refine_.emplace(i1, i0);
            }
            break;
        default:
            break;
    }

    return l0 - l1;
}

void Atlas::unlink(uuid id)
{
    m_adjacent_.erase(id);
    m_refine_.erase(id);
    m_coarsen_.erase(id);
    for (int i = 0; i < MAX_NUM_OF_LEVEL; ++i) { m_layer_[i].erase(id); }
}


void Atlas::update_all()
{
    for (auto ib = m_nodes_.begin(), ie = m_nodes_.end(); ib != ie; ++ib)
        for (auto it = ib; it != ie; ++it) { link(ib->first, it->first); }
    m_max_level_ = 0;
    for (unsigned int i = 0; i < MAX_NUM_OF_LEVEL; ++i)
    {
        if (m_layer_[i].size() == 0) { break; }
        m_max_level_ = i;
    }

}
//
//std::shared_ptr<TransitionMapBase>
//Atlas::add_adjacency(MeshBlockId first, MeshBlockId second)
//{
//    return add_adjacency(get_block(first), get_block(second));
//}
//
//std::shared_ptr<TransitionMapBase>
//Atlas::add_adjacency(std::shared_ptr<const MeshBase> first, std::shared_ptr<const MeshBase> second)
//{
//    return add_adjacency(std::make_shared<TransitionMapBase>(*first, *second));
//}
//
//std::tuple<std::shared_ptr<TransitionMapBase>, std::shared_ptr<TransitionMapBase>>
//Atlas::add_connection(std::shared_ptr<const MeshBase> first, std::shared_ptr<const MeshBase> second)
//{
//    return std::make_tuple(add_adjacency(first, second), add_adjacency(second, first));
//}


}}//namespace simpla{namespace mesh{