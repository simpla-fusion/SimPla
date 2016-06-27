/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "MeshAtlas.h"
#include "../gtl/BoxUtility.h"

namespace simpla { namespace mesh
{

TransitionMap::TransitionMap(Chart const *p_first, Chart const *p_second, int p_flag)
        : first(p_first), second(p_second), flag(p_flag),
          m_overlap_region_M_(gtl::box_overlap(first->box(SP_ES_LOCAL), second->box(SP_ES_OWNED)))
{

};

TransitionMap::~TransitionMap() { };

int TransitionMap::direct_pull_back(Real const *g, Real *f, mesh::MeshEntityType entity_type) const
{
    first->range(m_overlap_region_M_, entity_type).foreach(
            [&](mesh::MeshEntityId const &s)
            {
                f[first->hash(s)] = g[second->hash(direct_map(s))];
            });
};

MeshBlockId Atlas::add_block(std::shared_ptr<Chart> p_m)
{
    m_.emplace(std::make_pair(p_m->id(), p_m));
}

std::shared_ptr<Chart> Atlas::get_block(mesh::MeshBlockId m_id) const
{
    return m_.at(m_id);
}

void Atlas::remove_block(MeshBlockId const &m_id)
{
    UNIMPLEMENTED;
}

void Atlas::add_adjacency(mesh::MeshBlockId first, mesh::MeshBlockId second, int flag)
{
    UNIMPLEMENTED;
}

MeshBlockId Atlas::extent_block(mesh::MeshBlockId first_id, int const *offset_direction, size_type width)
{
    auto second_id = add_block(get_block(first_id)->extend(width, offset_direction));

    add_adjacency(first_id, second_id, SP_MB_SYNC);
    add_adjacency(second_id, first_id, SP_MB_SYNC);

    return second_id;

};


MeshBlockId Atlas::refine_block(mesh::MeshBlockId first, box_type const &)
{
    UNIMPLEMENTED;
}

MeshBlockId Atlas::coarsen_block(mesh::MeshBlockId first, box_type const &)
{
    UNIMPLEMENTED;
}

void Atlas::get_adjacencies(mesh::MeshBlockId first, int flag, std::list<std::shared_ptr<TransitionMap>> *res) const
{
    for (auto const &item:m_adjacency_list_.at(first))
    {
        if ((item->flag & flag) != 0x0)
        {
            assert(item->first->id() == first);

            res->push_back(item);
        }
    }

};


}}//namespace simpla{namespace mesh{