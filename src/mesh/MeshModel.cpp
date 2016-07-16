//
// Created by salmon on 16-6-2.
//

#include <set>
#include "../parallel/Parallel.h"
#include "MeshModel.h"
#include "MeshEntityId.h"

namespace simpla { namespace mesh
{

struct Model::pimpl_s
{

    struct MeshEntityIdHasher
    {
        int64_t operator()(const MeshEntityId &s) const { return s.v; }
        int64_t hash(const MeshEntityId &s) const { return s.v; }
    };

    typedef parallel::concurrent_hash_map<MeshEntityId, Real> cache_type;

    cache_type m_vertex_cache;
    typedef parallel::concurrent_hash_map<MeshEntityId, int> volume_cache_type;

    typedef parallel::concurrent_unordered_set<MeshEntityId> set_type;

    volume_cache_type m_volume_cache;

};

Model::Model(MeshBase const *pm)
    : m(pm), m_pimpl_(new pimpl_s) { }

Model::~Model() { }

void Model::add(MeshEntityRange const &r, distance_fun_t const distance)
{

    r.foreach(
        [&](MeshEntityId const &s)
        {
            auto x = m->point(s);

            Real d = distance(x);

            typename pimpl_s::cache_type::accessor acc;

            if (!(m_pimpl_->m_vertex_cache.insert(acc, s))) { acc->second = std::min(-d, acc->second); }
        }
    );

}
void Model::remove(MeshEntityRange const &r, distance_fun_t const distance)
{

    r.foreach(
        [&](MeshEntityId const &s)
        {
            auto x = m->point(s);

            Real d = distance(x);

            typename pimpl_s::cache_type::accessor acc;

            if (!(m_pimpl_->m_vertex_cache.insert(acc, s))) { acc->second = std::max(d, acc->second); }
        }
    );

}
void Model::deploy()
{
    m->range(VOLUME).foreach(
        [&](MeshEntityId const &s)
        {
            typename pimpl_s::volume_cache_type::accessor acc;
            m_pimpl_->m_volume_cache.insert(acc, s);
            acc->second = check(s);
        }
    );
};
int Model::check(MeshEntityId const &s)
{
    MeshEntityId p[MeshEntityIdCoder::MAX_NUM_OF_NEIGHBOURS];

    int num = m->get_adjacent_entities(VERTEX, s, p);

    int num_of_inside_vertex = 0;

    for (int i = 0; i < num; ++i)
    {
        typename pimpl_s::cache_type::const_accessor acc;

        if (m_pimpl_->m_vertex_cache.find(acc, p[i])) { if (acc->second <= 0.0) { ++num_of_inside_vertex; }}
    }

    int res = INSIDE;

    if (num_of_inside_vertex == num) { res = INSIDE; }
    else if (num_of_inside_vertex == 0) { res = OUTSIDE; }
    else if (num_of_inside_vertex > 0 && num_of_inside_vertex < num) { res = ON_SURFACE; }
    return res;
};
/**
 *  id < 0 out of surface
 *       = 0 on surface
 *       > 0 in surface
 */
MeshEntityRange  Model::surface(MeshEntityType iform, int flag)
{
    auto res_holder = MeshEntityRange::create<typename pimpl_s::set_type>();
    typename pimpl_s::set_type &res = res_holder.as<typename pimpl_s::set_type>();

    switch (iform)
    {
        case EDGE:
        case FACE:
            for (auto const &v_item:m_pimpl_->m_volume_cache)
            {
                if (v_item.second == ON_SURFACE)
                {
                    MeshEntityId p[MeshEntityIdCoder::MAX_NUM_OF_NEIGHBOURS];

                    int num =
                        m->get_adjacent_entities(iform, *reinterpret_cast<MeshEntityId const *>(&(v_item.first)), p);

                    for (int i = 0; i < num; ++i)
                    {
                        if ((check(p[i]) & flag) != 0) { res.insert(INT_2_ENTITY_ID(v_item.first)); }
                    }
                }
            }
            break;
        case VOLUME:
            for (auto const &v_item:m_pimpl_->m_volume_cache)
            {
                if (v_item.second == ON_SURFACE) { res.insert(INT_2_ENTITY_ID(v_item.first)); }
            }
            break;
        default: // case VERTEX:
            break;
    }
    return std::move(res_holder);

}

MeshEntityRange Model::inside(MeshEntityType iform)
{
    auto res_holder = MeshEntityRange::create<typename pimpl_s::set_type>();
    typename pimpl_s::set_type &res = res_holder.as<typename pimpl_s::set_type>();

    switch (iform)
    {
        case VERTEX:
            for (auto const &v_item:m_pimpl_->m_vertex_cache)
            {
                if (v_item.second <= 0)
                {
                    res.insert(INT_2_ENTITY_ID(v_item.first));
                }
            }
            break;
        case EDGE:
        case FACE:
            m->range(iform).foreach([&](MeshEntityId const &s) { if (check(s) == INSIDE) { res.insert(s); }});
            break;
        default:// case VOLUME:
            for (auto const &v_item:m_pimpl_->m_volume_cache)
            {
                if (v_item.second == INSIDE) { res.insert(INT_2_ENTITY_ID(v_item.first)); }
            }
            break;

    }
    return std::move(res_holder);
}
MeshEntityRange Model::outside(MeshEntityType iform)
{
    auto res_holder = MeshEntityRange::create<typename pimpl_s::set_type>();
    typename pimpl_s::set_type &res = res_holder.as<typename pimpl_s::set_type>();

    switch (iform)
    {
        case VERTEX:
            for (auto const &v_item:m_pimpl_->m_vertex_cache)
            {
                if (v_item.second > 0)
                {
                    res.insert(INT_2_ENTITY_ID(v_item.first));
                }
            }
            break;
        case EDGE:
        case FACE:
            m->range(iform).foreach([&](MeshEntityId const &s) { if (check(s) == OUTSIDE) { res.insert(s); }});
            break;
        default:// case VOLUME:
            for (auto const &v_item:m_pimpl_->m_volume_cache)
            {
                if (v_item.second == OUTSIDE) { res.insert(INT_2_ENTITY_ID(v_item.first)); }
            }
            break;

    }
    return std::move(res_holder);
}

}}//namespace simpla{namespace get_mesh{