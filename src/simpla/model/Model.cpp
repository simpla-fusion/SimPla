//
// Created by salmon on 16-6-2.
//

#include <set>
#include <simpla/mesh/EntityId.h>
#include <simpla/mesh/EntityIdRange.h>
#include <simpla/manifold/Chart.h>
#include <simpla/toolbox/Parallel.h>
#include <simpla/mesh/MeshCommon.h>
#include "Model.h"

namespace simpla { namespace model
{
using namespace mesh;


Model::Model(std::shared_ptr<Chart> const &c) : m_chart_(c) {}

Model::~Model() {}

void Model::load(std::string const &) { UNIMPLEMENTED; }

void Model::save(std::string const &) { UNIMPLEMENTED; }

std::ostream &Model::print(std::ostream &os, int indent) const { return os; }

void Model::deploy() {};

void Model::pre_process()
{
    m_tags_.pre_process();
    m_tags_.clear();
};

void Model::initialize(Real data_time)
{

    auto m_start_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 9> *>(m_tags_.data_block())->start();
    auto m_count_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 9> *>(m_tags_.data_block())->count();

    index_type ib = m_start_[0];
    index_type ie = m_start_[0] + m_count_[0];
    index_type jb = m_start_[1];
    index_type je = m_start_[1] + m_count_[1];
    index_type kb = m_start_[2];
    index_type ke = m_start_[2] + m_count_[2];


    for (index_type i = ib; i < ie; ++i)
        for (index_type j = jb; j < je; ++j)
            for (index_type k = kb; k < ke; ++k)
            {
                auto x = m_chart_->mesh_block()->point(i, j, k);
                auto &tag = m_tags_.get(i, j, k, 0);

                tag = VACUUME;

                for (auto const &obj:m_g_obj_) { if (obj.second->check_inside(x)) { tag |= obj.first; }}

            }
    for (index_type i = ib; i < ie - 1; ++i)
        for (index_type j = jb; j < je - 1; ++j)
            for (index_type k = kb; k < ke - 1; ++k)
            {
                m_tags_.get(i, j, k, 1) = m_tags_.get(i, j, k, 0) | m_tags_.get(i + 1, j, k, 0);
                m_tags_.get(i, j, k, 2) = m_tags_.get(i, j, k, 0) | m_tags_.get(i, j + 1, k, 0);
                m_tags_.get(i, j, k, 4) = m_tags_.get(i, j, k, 0) | m_tags_.get(i, j, k + 1, 0);


            }


    for (index_type i = ib; i < ie - 1; ++i)
        for (index_type j = jb; j < je - 1; ++j)
            for (index_type k = kb; k < ke - 1; ++k)
            {
                m_tags_.get(i, j, k, 3) = m_tags_.get(i, j, k, 1) | m_tags_.get(i, j + 1, k, 1);
                m_tags_.get(i, j, k, 5) = m_tags_.get(i, j, k, 1) | m_tags_.get(i, j, k + 1, 1);
                m_tags_.get(i, j, k, 6) = m_tags_.get(i, j + 1, k, 1) | m_tags_.get(i, j, k + 1, 1);
            }

    for (index_type i = ib; i < ie - 1; ++i)
        for (index_type j = jb; j < je - 1; ++j)
            for (index_type k = kb; k < ke - 1; ++k)
            {
                m_tags_.get(i, j, k, 7) = m_tags_.get(i, j, k, 3) | m_tags_.get(i, j, k + 1, 3);
            }

};

void Model::next_time_step(Real data_time, Real dt) {};

void Model::finalize(Real data_time)
{
    m_range_cache_.erase(m_chart_->mesh_block()->id());
    m_interface_cache_.erase(m_chart_->mesh_block()->id());
};

void Model::post_process() {};


void Model::add_object(std::string const &key, std::shared_ptr<geometry::GeoObject> const &g_obj)
{

    int id = 0;
    auto it = m_g_name_map_.find(key);
    if (it != m_g_name_map_.end()) { id = it->second; }
    else
    {
        id = 1 << m_g_obj_count_;
        ++m_g_obj_count_;
        m_g_name_map_[key] = id;
    }
    m_g_obj_.insert(std::make_pair(id, g_obj));
}


void Model::remove_object(std::string const &key)
{
    try
    {
        m_g_obj_.erase(m_g_name_map_.at(key));
    } catch (...)
    {

    }
}

mesh::EntityIdRange const &
Model::select(MeshEntityType iform, std::string const &tag)
{
    return select(iform, m_g_name_map_.at(tag));
}


mesh::EntityIdRange const &
Model::select(MeshEntityType iform, int tag)
{

    typedef mesh::MeshEntityIdCoder M;
    typedef parallel::concurrent_unordered_set<MeshEntityId, MeshEntityIdHasher> set_type;

    try { return m_range_cache_.at(iform).at(tag); } catch (...) {}

    const_cast<this_type *>(this)->m_range_cache_[iform].
            emplace(std::make_pair(tag, EntityIdRange::create<set_type>()));

    auto &res = m_range_cache_.at(iform).at(tag).as<set_type>();

    auto m_start_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 9> *>(m_tags_.data_block())->start();
    auto m_count_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 9> *>(m_tags_.data_block())->count();

    index_type ib = m_start_[0];
    index_type ie = m_start_[0] + m_count_[0];
    index_type jb = m_start_[1];
    index_type je = m_start_[1] + m_count_[1];
    index_type kb = m_start_[2];
    index_type ke = m_start_[2] + m_count_[2];

#define _CAS(I, J, K, L) if (I>=0 && J>=0 && K>=0 && ((m_tags_.get(I, J, K, L) &tag) == tag)) { res.insert(M::pack_index(I, J, K, L)); }

    switch (iform)
    {
        case VERTEX:
#pragma omp parallel for
            for (index_type i = ib; i < ie; ++i)
                for (index_type j = jb; j < je; ++j)
                    for (index_type k = kb; k < ke; ++k)
                    {
                        _CAS(i, j, k, 0);
                    }

            break;
        case EDGE:
#pragma omp parallel for
            for (index_type i = ib; i < ie - 1; ++i)
                for (index_type j = jb; j < je - 1; ++j)
                    for (index_type k = kb; k < ke - 1; ++k)
                    {
                        _CAS(i, j, k, 1);
                        _CAS(i, j, k, 2);
                        _CAS(i, j, k, 4);
                    }
            break;
        case FACE:
#pragma omp parallel for
            for (index_type i = ib; i < ie - 1; ++i)
                for (index_type j = jb; j < je - 1; ++j)
                    for (index_type k = kb; k < ke - 1; ++k)
                    {
                        _CAS(i, j, k, 3);
                        _CAS(i, j, k, 5);
                        _CAS(i, j, k, 6);
                    }
            break;
        case VOLUME:
#pragma omp parallel for
            for (index_type i = ib; i < ie - 1; ++i)
                for (index_type j = jb; j < je - 1; ++j)
                    for (index_type k = kb; k < ke - 1; ++k)
                    {
                        _CAS(i, j, k, 7);
                    }
            break;
        default:
            break;
    }
#undef _CAS
    return m_range_cache_.at(iform).at(tag);;

}

/**
 *  id < 0 out of surface
 *       = 0 on surface
 *       > 0 in surface
 */
mesh::EntityIdRange const &Model::interface(MeshEntityType iform, const std::string &s_in, const std::string &s_out)
{
    return interface(iform, m_g_name_map_.at(s_in), m_g_name_map_.at(s_out));
}

mesh::EntityIdRange const &Model::interface(MeshEntityType iform, int tag_in, int tag_out)
{

    try { return m_interface_cache_.at(iform).at(tag_in).at(tag_out); } catch (...) {}

    typedef mesh::MeshEntityIdCoder M;

    typedef parallel::concurrent_unordered_set<MeshEntityId, MeshEntityIdHasher> set_type;

    const_cast<this_type *>(this)->m_interface_cache_[iform][tag_in].
            emplace(std::make_pair(tag_out, EntityIdRange::create<set_type>()));

    set_type &res = const_cast<this_type *>(this)->m_interface_cache_.at(iform).at(tag_in).at(tag_out).as<set_type>();

    auto m_start_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 9> *>(m_tags_.data_block())->start();
    auto m_count_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 9> *>(m_tags_.data_block())->count();

    index_type ib = m_start_[0];
    index_type ie = m_start_[0] + m_count_[0];
    index_type jb = m_start_[1];
    index_type je = m_start_[1] + m_count_[1];
    index_type kb = m_start_[2];
    index_type ke = m_start_[2] + m_count_[2];

    int v_tag = tag_in | tag_out;
#pragma omp parallel for
    for (index_type i = ib; i < ie - 1; ++i)
        for (index_type j = jb; j < je - 1; ++j)
            for (index_type k = kb; k < ke - 1; ++k)
            {
                if ((m_tags_.get(i, j, k, 7) & v_tag) != v_tag) { continue; }
#define _CAS(I, J, K, L) if (I>=0 && J>=0 && K>=0 && m_tags_.get(I, J, K, L)  == tag_in) { res.insert(M::pack_index(I, J, K, L)); }
                switch (iform)
                {
                    case VERTEX:
                        _CAS(i + 0, j + 0, k + 0, 0);
                        _CAS(i + 1, j + 0, k + 0, 0);
                        _CAS(i + 0, j + 1, k + 0, 0);
                        _CAS(i + 1, j + 1, k + 0, 0);
                        _CAS(i + 0, j + 0, k + 1, 0);
                        _CAS(i + 1, j + 0, k + 1, 0);
                        _CAS(i + 0, j + 1, k + 1, 0);
                        _CAS(i + 1, j + 1, k + 1, 0);


                        break;
                    case EDGE:
                        _CAS(i + 0, j + 0, k + 0, 1);
                        _CAS(i + 0, j + 1, k + 0, 1);
                        _CAS(i + 0, j + 0, k + 1, 1);
                        _CAS(i + 0, j + 1, k + 1, 1);

                        _CAS(i + 0, j + 0, k + 0, 2);
                        _CAS(i + 1, j + 0, k + 0, 2);
                        _CAS(i + 0, j + 0, k + 1, 2);
                        _CAS(i + 1, j + 0, k + 1, 2);

                        _CAS(i + 0, j + 0, k + 0, 4);
                        _CAS(i + 0, j + 1, k + 0, 4);
                        _CAS(i + 1, j + 0, k + 0, 4);
                        _CAS(i + 1, j + 1, k + 0, 4);
                        break;
                    case FACE:
                        _CAS(i + 0, j + 0, k + 0, 3);
                        _CAS(i + 0, j + 0, k + 1, 3);

                        _CAS(i + 0, j + 0, k + 0, 5);
                        _CAS(i + 0, j + 1, k + 0, 5);

                        _CAS(i + 0, j + 0, k + 0, 6);
                        _CAS(i + 0, j + 0, k + 1, 6);
                        break;
                    case VOLUME:
                        _CAS(i - 1, j - 1, k - 1, 7);
                        _CAS(i + 0, j - 1, k - 1, 7);
                        _CAS(i + 1, j - 1, k - 1, 7);
                        _CAS(i - 1, j + 0, k - 1, 7);
                        _CAS(i + 0, j + 0, k - 1, 7);
                        _CAS(i + 1, j + 0, k - 1, 7);
                        _CAS(i - 1, j + 1, k - 1, 7);
                        _CAS(i + 0, j + 1, k - 1, 7);
                        _CAS(i + 1, j + 1, k - 1, 7);


                        _CAS(i - 1, j - 1, k + 0, 7);
                        _CAS(i + 0, j - 1, k + 0, 7);
                        _CAS(i + 1, j - 1, k + 0, 7);
                        _CAS(i - 1, j + 0, k + 0, 7);
                        //   _CAS(i + 0, j + 0, k + 0, 7);
                        _CAS(i + 1, j + 0, k + 0, 7);
                        _CAS(i - 1, j + 1, k + 0, 7);
                        _CAS(i + 0, j + 1, k + 0, 7);
                        _CAS(i + 1, j + 1, k + 0, 7);


                        _CAS(i - 1, j - 1, k + 1, 7);
                        _CAS(i + 0, j - 1, k + 1, 7);
                        _CAS(i + 1, j - 1, k + 1, 7);
                        _CAS(i - 1, j + 0, k + 1, 7);
                        _CAS(i + 0, j + 0, k + 1, 7);
                        _CAS(i + 1, j + 0, k + 1, 7);
                        _CAS(i - 1, j + 1, k + 1, 7);
                        _CAS(i + 0, j + 1, k + 1, 7);
                        _CAS(i + 1, j + 1, k + 1, 7);
                        break;
                    default:
                        break;
                }
#undef _CAS

            }


    return m_interface_cache_.at(iform).at(tag_in).at(tag_out);

}


}}//namespace simpla{namespace get_mesh{