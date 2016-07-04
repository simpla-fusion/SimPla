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
          m_overlap_region_M_(gtl::box_overlap(first->box(SP_ES_ALL), second->box(SP_ES_OWNED)))
{

    auto b_first = p_first->index_box(m_overlap_region_M_);
    auto b_second = p_second->index_box(m_overlap_region_M_);

    index_tuple offset;
    offset = std::get<0>(b_second) - std::get<0>(b_first);
    m_offset_ = MeshEntityIdCoder::pack_index(offset);
//    if (p_first->name() == "PML_3")
//    {
//        VERBOSE << p_second->name() << "\t" << m_overlap_region_M_ << b_first << b_second << offset << std::endl;
//    }



//    m_offset_.v = (std::get<0>(second->point_global_to_local(std::get<0>(m_overlap_region_M_)))
//                   - std::get<0>(first->point_global_to_local(std::get<0>(m_overlap_region_M_)))
//                   - MeshEntityIdCoder::_DA - MeshEntityIdCoder::_DA).v & ~MeshEntityIdCoder::_DA.v;

};

TransitionMap::TransitionMap(std::shared_ptr<Chart const> m, std::shared_ptr<Chart const> n, int flag) :
        TransitionMap(m.get(), n.get(), flag) { }

TransitionMap::~TransitionMap() { };


int TransitionMap::direct_pull_back(void *f, void const *g, size_type ele_size_in_byte,
                                    MeshEntityType entity_type) const
{
//    first->range(m_overlap_region_M_, entity_type).foreach(
//            [&](mesh::MeshEntityId const &s)
//            {
//                memcpy(reinterpret_cast<byte_type *>(f) + first->hash(s) * ele_size_in_byte,
//                       reinterpret_cast<byte_type const *>(g) + second->hash(direct_map(s)) * ele_size_in_byte,
//                       ele_size_in_byte
//                );
//
//            });
}


int TransitionMap::map(point_type *x) const { return 1; };

point_type TransitionMap::map(point_type const &x) const { return x; }

mesh::MeshEntityId TransitionMap::direct_map(mesh::MeshEntityId s) const
{
    return s + m_offset_;
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

void Atlas::add_adjacency(std::shared_ptr<mesh::MeshBase> first, std::shared_ptr<mesh::MeshBase> second, int flag)
{
    m_adjacency_list_.emplace(std::make_pair(first->id(), std::make_shared<TransitionMap>(
            first, second, flag)));
}

void Atlas::add_adjacency_2(std::shared_ptr<mesh::MeshBase> first, std::shared_ptr<mesh::MeshBase> second, int flag)
{
    add_adjacency(first, second, flag);
    add_adjacency(second, first, flag);

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