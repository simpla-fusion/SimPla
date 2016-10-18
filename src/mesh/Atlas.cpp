/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "Atlas.h"
#include "TransitionMap.h"

#include "../toolbox/BoxUtility.h"

namespace simpla { namespace mesh
{

MeshBlockId
Atlas::add_block(std::shared_ptr<MeshBase> p_m)
{
    auto res = m_nodes_.emplace(p_m->id(), p_m);
    return res.first->first;
}

std::shared_ptr<MeshBase>
Atlas::get_block(mesh::MeshBlockId m_id) const
{
    assert(m_nodes_.at(m_id) != nullptr);
    return m_nodes_.at(m_id);
}

std::shared_ptr<TransitionMapBase>
Atlas::add_adjacency(std::shared_ptr<TransitionMapBase> t_map)
{

    m_out_edge_.emplace(t_map->from_id(), t_map);
    m_in_edge_.emplace(t_map->to_id(), t_map);
    return t_map;
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