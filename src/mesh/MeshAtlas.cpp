/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "MeshAtlas.h"
#include "../gtl/BoxUtility.h"

namespace simpla { namespace mesh
{

TransitionMap::TransitionMap(Chart const *p_first, Chart const *p_second, index_box_type i_box, index_tuple offset,
                             int p_flag)
        : first(p_first), second(p_second), m_overlap_idx_box_(i_box), m_offset_(MeshEntityIdCoder::pack_index(offset)),
          flag(p_flag)
{
};


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
    m_list_.emplace(std::make_pair(p_m->id(), p_m));
    return p_m->id();
}

std::shared_ptr<Chart> Atlas::get_block(mesh::MeshBlockId m_id) const
{
    assert(m_list_.at(m_id) != nullptr);
    return m_list_.at(m_id);
}

void Atlas::add_adjacency(std::shared_ptr<TransitionMap> t_map)
{
    m_adjacency_list_.emplace(std::make_pair(t_map->first->id(), t_map));
}

std::shared_ptr<TransitionMap> Atlas::add_adjacency(MeshBlockId first, MeshBlockId second, int flag)
{
    return add_adjacency(get_block(first).get(), get_block(second).get(), flag);
}

std::shared_ptr<TransitionMap> Atlas::add_adjacency(const MeshBase *first, const MeshBase *second, int flag)
{
    box_type x_b_first = gtl::box_overlap(first->box(SP_ES_ALL), second->box(SP_ES_OWNED));
    index_box_type i_b_first = first->index_box(x_b_first);
    index_box_type i_b_second = second->index_box(x_b_first);
    index_tuple offset;

    offset = std::get<0>(i_b_second) - std::get<0>(i_b_first);
    auto res = std::make_shared<TransitionMap>(first, second, i_b_first, offset, flag);
    add_adjacency(res);
    return res;
}

void Atlas::add_adjacency2(const MeshBase *first, const MeshBase *second, int flag)
{
    auto t0 = add_adjacency(first, second, flag);
    auto t1 = add_adjacency(second, first, flag);

    CHECK(1) << t0->m_overlap_idx_box_ << MeshEntityIdCoder::unpack_index(t0->m_offset_);
    CHECK(2) << t1->m_overlap_idx_box_ << MeshEntityIdCoder::unpack_index(t1->m_offset_);;

}


io::IOStream &Atlas::save(io::IOStream &os) const
{

    for (auto const &item:m_list_)
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