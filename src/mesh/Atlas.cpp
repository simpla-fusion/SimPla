/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "Atlas.h"
#include "TransitionMap.h"

#include "../toolbox/BoxUtility.h"

namespace simpla { namespace mesh
{

MeshBlockId Atlas::add_block(std::shared_ptr<Block> p_m)
{
    m_nodes_.emplace(std::make_pair(p_m->id(), p_m));
    return p_m->id();
}

std::shared_ptr<Block> Atlas::get_block(mesh::MeshBlockId m_id) const
{
    assert(m_nodes_.at(m_id) != nullptr);
    return m_nodes_.at(m_id);
}

void Atlas::add_adjacency(std::shared_ptr<TransitionMap> t_map)
{
    m_adjacency_list_.emplace(std::make_pair(t_map->m_dst_->id(), t_map));
}

std::shared_ptr<TransitionMap> Atlas::add_adjacency(MeshBlockId first, MeshBlockId second)
{
    return add_adjacency(get_block(first), get_block(second));
}

std::shared_ptr<TransitionMap>
Atlas::add_adjacency(std::shared_ptr<const Block> first, std::shared_ptr<const Block> second)
{
    auto res = std::make_shared<TransitionMap>(*first, *second);
    add_adjacency(res);
    return res;
}

void Atlas::add_adjacency2(std::shared_ptr<const Block> first, std::shared_ptr<const Block> second)
{
    auto t0 = add_adjacency(first, second);
    auto t1 = add_adjacency(second, first);
//
//    CHECK(1) << t0->m_overlap_idx_box_ << MeshEntityIdCoder::unpack_index(t0->m_offset_);
//    CHECK(2) << t1->m_overlap_idx_box_ << MeshEntityIdCoder::unpack_index(t1->m_offset_);;

}


}}//namespace simpla{namespace mesh{