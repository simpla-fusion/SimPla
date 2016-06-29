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


int TransitionMap::direct_pull_back(void *f, void const *g, size_type ele_size_in_byte,
                                    MeshEntityType entity_type) const
{
    first->range(m_overlap_region_M_, entity_type).foreach(
            [&](mesh::MeshEntityId const &s)
            {
                memcpy(reinterpret_cast<byte_type *>(f) + first->hash(s) * ele_size_in_byte,
                       reinterpret_cast<byte_type const *>(g) + second->hash(direct_map(s)) * ele_size_in_byte,
                       ele_size_in_byte
                );

            });
}


int TransitionMap::map(point_type *x) const { return 1; };

point_type TransitionMap::map(point_type const &x) const { return x; }

mesh::MeshEntityId TransitionMap::direct_map(mesh::MeshEntityId s) const
{
    return s;
};

MeshBlockId Atlas::add_block(std::shared_ptr<Chart> p_m)
{
    m_.emplace(std::make_pair(p_m->id(), p_m));
    return p_m->id();
}

std::shared_ptr<Chart> Atlas::get_block(mesh::MeshBlockId m_id) const
{
    assert(m_.at(m_id) != nullptr);
    return m_.at(m_id);
}

void Atlas::add_adjacency(mesh::MeshBlockId first, mesh::MeshBlockId second, int flag)
{
    m_adjacency_list_.emplace(std::make_pair(first, std::make_shared<TransitionMap>(
            get_block(first).get(), get_block(second).get(), flag)));
}


io::IOStream &Atlas::save(io::IOStream &os) const
{

    for (auto const &item:m_)
    {
        item.second->save(os);
    }
    return os;
}

io::IOStream &Atlas::load(io::IOStream &is)
{
    UNIMPLEMENTED;
    return is;
}


}}//namespace simpla{namespace mesh{