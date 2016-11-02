/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "Atlas.h"
#include "TransitionMap.h"

#include "simpla/toolbox/BoxUtility.h"

namespace simpla { namespace mesh
{

struct Atlas::pimpl_s
{
    typedef typename std::multimap<id_type, id_type>::iterator link_iterator;
    typedef typename std::multimap<id_type, id_type>::const_iterator const_link_iterator;
    typedef std::pair<const_link_iterator, const_link_iterator> multi_links_type;
    std::map<id_type, std::shared_ptr<MeshBlock>> m_nodes_;
    std::multimap<id_type, id_type> m_adjacent_;
    std::multimap<id_type, id_type> m_refine_;
    std::multimap<id_type, id_type> m_coarsen_;
    std::set<id_type> m_layer_[MAX_NUM_OF_LEVEL];
    unsigned int m_max_level_ = 0;


    std::set<std::shared_ptr<AttributeBase>> m_attrs_;
};

Atlas::Atlas() : m_pimpl_(new pimpl_s) {};

Atlas::~Atlas() {};

bool Atlas::has(id_type id) const { return m_pimpl_->m_nodes_.find(id) != m_pimpl_->m_nodes_.end(); };

MeshBlock &Atlas::at(id_type id)
{
    id = (id == 0) ? m_pimpl_->m_nodes_.begin()->first : id;
    return *(m_pimpl_->m_nodes_.at(id));
};

MeshBlock const &Atlas::at(id_type id) const
{
    id = (id == 0) ? m_pimpl_->m_nodes_.begin()->first : id;
    return *(m_pimpl_->m_nodes_.at(id));
};

id_type Atlas::insert(std::shared_ptr<MeshBlock> const p_m, id_type hint)
{
    m_pimpl_->m_nodes_.emplace(std::make_pair(p_m->id(), p_m));
    link(p_m->id(), hint);
    return p_m->id();
};


void Atlas::erase(id_type id)
{
    if (!has(id)) { return; }

    unlink(id);

    for (auto &item:m_pimpl_->m_attrs_) { item->erase(id); }

    m_pimpl_->m_nodes_.erase(id);

}

void Atlas::update(id_type id)
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
    MeshBlock const &m = *(it->second);
    assert(m.level() < MAX_NUM_OF_LEVEL);
    m_max_level_ = std::max(m_max_level_, m.level());
    m_layer_[m.level()].insert(id);

    for (auto const &item:m_nodes_) { if (item.first != id) { link(id, item.first); }}
}

int Atlas::link(id_type i0, id_type i1)
{
    assert(has(i0) && has(i1));
    MeshBlock const &m0 = *at(i0);
    MeshBlock const &m1 = *at(i1);
    int l0 = m0.level();
    int l1 = m1.level();
    box_type b0 = m0.box();
    box_type b1 = m1.box();
    vector_type dx = m0.dx();
    vector_type L;//= m0.period_length();
    switch (l0 - l1)
    {
        case 0:
            if (toolbox::check_adjoining(b0, b1, dx, L))
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

void Atlas::unlink(id_type id)
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

void Atlas::register_attribute(std::shared_ptr<AttributeBase> attr)
{

}

void Atlas::unregister_attribute(std::shared_ptr<AttributeBase> attr)
{

}
void Atlas::clear(id_type id){

};

void Atlas::sync(id_type dest, id_type src){

};

void Atlas::coarsen(id_type id){

};

void Atlas::refine(id_type id){

};

void Atlas::deploy(id_type id ){

};
//
//std::shared_ptr<TransitionMapBase>
//Atlas::add_adjacency(MeshBlockId first, MeshBlockId second)
//{
//    return add_adjacency(get_block(first), get_block(second));
//}
//
//std::shared_ptr<TransitionMapBase>
//Atlas::add_adjacency(std::shared_ptr<const MeshBlock> first, std::shared_ptr<const MeshBlock> second)
//{
//    return add_adjacency(std::make_shared<TransitionMapBase>(*first, *second));
//}
//
//std::tuple<std::shared_ptr<TransitionMapBase>, std::shared_ptr<TransitionMapBase>>
//Atlas::add_connection(std::shared_ptr<const MeshBlock> first, std::shared_ptr<const MeshBlock> second)
//{
//    return std::make_tuple(add_adjacency(first, second), add_adjacency(second, first));
//}


}}//namespace simpla{namespace mesh_as{