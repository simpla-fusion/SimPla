//
// Created by salmon on 16-6-2.
//

#include "Model.h"
#include <simpla/engine/AttributeView.h>
#include <simpla/engine/MeshBlock.h>
#include <simpla/mesh/EntityId.h>
namespace simpla {
namespace model {
using namespace mesh;
using namespace engine;

Model::Model() {}

Model::~Model() {}

void Model::Load(std::string const&) { UNIMPLEMENTED; }

void Model::Save(std::string const&) { UNIMPLEMENTED; }

std::ostream& Model::Print(std::ostream& os, int indent) const { return os; }

void Model::Deploy(){};

void Model::PreProcess() {
    m_tags_.Update();
    m_tags_.Clear();
};

void Model::Initialize(Real data_time, Real dt) {
    PreProcess();
    //
    //    index_type const* lower = m_tags_.lower();
    //    index_type const* upper = m_tags_.upper();
    //
    //    index_type ib = lower[0];
    //    index_type ie = upper[0];
    //    index_type jb = lower[1];
    //    index_type je = upper[1];
    //    index_type kb = lower[2];
    //    index_type ke = upper[2];
    //
    //    for (index_type i = ib; i < ie; ++i)
    //        for (index_type j = jb; j < je; ++j)
    //            for (index_type k = kb; k < ke; ++k) {
    //                auto x = m_mesh_->mesh_block()->point(i, j, k);
    //                auto& tag = m_tags_(i, j, k, 0);
    //
    //                tag = VACUUM;
    //
    //                for (auto const& obj : m_g_obj_) {
    //                    if (obj.second->check_inside(x)) { tag |= obj.first; }
    //                }
    //            }
    //    for (index_type i = ib; i < ie - 1; ++i)
    //        for (index_type j = jb; j < je - 1; ++j)
    //            for (index_type k = kb; k < ke - 1; ++k) {
    //                m_tags_(i, j, k, 1) = m_tags_(i, j, k, 0) | m_tags_(i + 1, j, k, 0);
    //                m_tags_(i, j, k, 2) = m_tags_(i, j, k, 0) | m_tags_(i, j + 1, k, 0);
    //                m_tags_(i, j, k, 4) = m_tags_(i, j, k, 0) | m_tags_(i, j, k + 1, 0);
    //            }
    //
    //    for (index_type i = ib; i < ie - 1; ++i)
    //        for (index_type j = jb; j < je - 1; ++j)
    //            for (index_type k = kb; k < ke - 1; ++k) {
    //                m_tags_(i, j, k, 3) = m_tags_(i, j, k, 1) | m_tags_(i, j + 1, k, 1);
    //                m_tags_(i, j, k, 5) = m_tags_(i, j, k, 1) | m_tags_(i, j, k + 1, 1);
    //                m_tags_(i, j, k, 6) = m_tags_(i, j + 1, k, 1) | m_tags_(i, j, k + 1, 1);
    //            }
    //
    //    for (index_type i = ib; i < ie - 1; ++i)
    //        for (index_type j = jb; j < je - 1; ++j)
    //            for (index_type k = kb; k < ke - 1; ++k) {
    //                m_tags_(i, j, k, 7) = m_tags_(i, j, k, 3) | m_tags_(i, j, k + 1, 3);
    //            }
};

void Model::NextTimeStep(Real data_time, Real dt){};

void Model::Finalize(Real data_time, Real dt) {
    m_range_cache_.erase(m_mesh_->mesh_block()->id());
    m_interface_cache_.erase(m_mesh_->mesh_block()->id());
    PostProcess();
};

void Model::PostProcess(){};

void Model::AddObject(std::string const& key, std::shared_ptr<geometry::GeoObject> const& g_obj) {
    int id = 0;
    auto it = m_g_name_map_.find(key);
    if (it != m_g_name_map_.end()) {
        id = it->second;
    } else {
        id = 1 << m_g_obj_count_;
        ++m_g_obj_count_;
        m_g_name_map_[key] = id;
    }
    m_g_obj_.insert(std::make_pair(id, g_obj));
}

void Model::RemoveObject(std::string const& key) {
    try {
        m_g_obj_.erase(m_g_name_map_.at(key));
    } catch (...) {}
}

Range<id_type> const& Model::select(int iform, std::string const& tag) { return select(iform, m_g_name_map_.at(tag)); }

Range<id_type> const& Model::select(int iform, int tag) {
    //    typedef MeshEntityIdCoder M;
    //
    //    try {
    //        return m_range_cache_.at(iform).at(tag);
    //    } catch (...) {}
    //
    //    const_cast<this_type*>(this)->m_range_cache_[iform].emplace(
    //        std::make_pair(tag, Range<id_type>(std::make_shared<UnorderedRange<id_type>>())));
    //
    //    auto& res = *m_range_cache_.at(iform).at(tag).self().template as<UnorderedRange<id_type>>();
    //
    //    index_type const* lower = m_tags_.lower();
    //    index_type const* upper = m_tags_.upper();
    //
    //    index_type ib = lower[0];
    //    index_type ie = upper[0];
    //    index_type jb = lower[1];
    //    index_type je = upper[1];
    //    index_type kb = lower[2];
    //    index_type ke = upper[2];
    //
    //#define _CAS(I, J, K, L) \
//    if (I >= 0 && J >= 0 && K >= 0 && ((m_tags_(I, J, K, L) & tag) == tag)) { res.insert(M::pack_index(I, J, K, L)); }
    //
    //    switch (iform) {
    //        case VERTEX:
    //#pragma omp parallel for
    //            for (index_type i = ib; i < ie; ++i)
    //                for (index_type j = jb; j < je; ++j)
    //                    for (index_type k = kb; k < ke; ++k) { _CAS(i, j, k, 0); }
    //
    //            break;
    //        case EDGE:
    //#pragma omp parallel for
    //            for (index_type i = ib; i < ie - 1; ++i)
    //                for (index_type j = jb; j < je - 1; ++j)
    //                    for (index_type k = kb; k < ke - 1; ++k) {
    //                        _CAS(i, j, k, 1);
    //                        _CAS(i, j, k, 2);
    //                        _CAS(i, j, k, 4);
    //                    }
    //            break;
    //        case FACE:
    //#pragma omp parallel for
    //            for (index_type i = ib; i < ie - 1; ++i)
    //                for (index_type j = jb; j < je - 1; ++j)
    //                    for (index_type k = kb; k < ke - 1; ++k) {
    //                        _CAS(i, j, k, 3);
    //                        _CAS(i, j, k, 5);
    //                        _CAS(i, j, k, 6);
    //                    }
    //            break;
    //        case VOLUME:
    //#pragma omp parallel for
    //            for (index_type i = ib; i < ie - 1; ++i)
    //                for (index_type j = jb; j < je - 1; ++j)
    //                    for (index_type k = kb; k < ke - 1; ++k) { _CAS(i, j, k, 7); }
    //            break;
    //        default:
    //            break;
    //    }
    //#undef _CAS
    //    return m_range_cache_.at(iform).at(tag);
    //    ;
}

/**
 *  id < 0 out of surface
 *       = 0 on surface
 *       > 0 in surface
 */
Range<id_type> const& Model::interface(int iform, const std::string& s_in, const std::string& s_out) {
    return interface(iform, m_g_name_map_.at(s_in), m_g_name_map_.at(s_out));
}

Range<id_type> const& Model::interface(int iform, int tag_in, int tag_out) {
//    try {
//        return m_interface_cache_.at(iform).at(tag_in).at(tag_out);
//    } catch (...) {}
//
//    typedef mesh::MeshEntityIdCoder M;
//
//    const_cast<this_type*>(this)->m_interface_cache_[iform][tag_in].emplace(
//        std::make_pair(tag_out, Range<id_type>(std::make_shared<UnorderedRange<id_type>>())));
//
//    auto& res = *const_cast<this_type*>(this)
//                     ->m_interface_cache_.at(iform)
//                     .at(tag_in)
//                     .at(tag_out)
//                     .self()
//                     .template as<UnorderedRange<id_type>>();
//
//    index_type const* lower = m_tags_.lower();
//    index_type const* upper = m_tags_.upper();
//
//    index_type ib = lower[0];
//    index_type ie = upper[0];
//    index_type jb = lower[1];
//    index_type je = upper[1];
//    index_type kb = lower[2];
//    index_type ke = upper[2];
//
//    int v_tag = tag_in | tag_out;
//#pragma omp parallel for
//    for (index_type i = ib; i < ie - 1; ++i)
//        for (index_type j = jb; j < je - 1; ++j)
//            for (index_type k = kb; k < ke - 1; ++k) {
//                if ((m_tags_(i, j, k, 7) & v_tag) != v_tag) { continue; }
//#define _CAS(I, J, K, L) \
//    if (I >= 0 && J >= 0 && K >= 0 && m_tags_(I, J, K, L) == tag_in) { res.insert(M::pack_index(I, J, K, L)); }
//                switch (iform) {
//                    case VERTEX:
//                        _CAS(i + 0, j + 0, k + 0, 0);
//                        _CAS(i + 1, j + 0, k + 0, 0);
//                        _CAS(i + 0, j + 1, k + 0, 0);
//                        _CAS(i + 1, j + 1, k + 0, 0);
//                        _CAS(i + 0, j + 0, k + 1, 0);
//                        _CAS(i + 1, j + 0, k + 1, 0);
//                        _CAS(i + 0, j + 1, k + 1, 0);
//                        _CAS(i + 1, j + 1, k + 1, 0);
//
//                        break;
//                    case EDGE:
//                        _CAS(i + 0, j + 0, k + 0, 1);
//                        _CAS(i + 0, j + 1, k + 0, 1);
//                        _CAS(i + 0, j + 0, k + 1, 1);
//                        _CAS(i + 0, j + 1, k + 1, 1);
//
//                        _CAS(i + 0, j + 0, k + 0, 2);
//                        _CAS(i + 1, j + 0, k + 0, 2);
//                        _CAS(i + 0, j + 0, k + 1, 2);
//                        _CAS(i + 1, j + 0, k + 1, 2);
//
//                        _CAS(i + 0, j + 0, k + 0, 4);
//                        _CAS(i + 0, j + 1, k + 0, 4);
//                        _CAS(i + 1, j + 0, k + 0, 4);
//                        _CAS(i + 1, j + 1, k + 0, 4);
//                        break;
//                    case FACE:
//                        _CAS(i + 0, j + 0, k + 0, 3);
//                        _CAS(i + 0, j + 0, k + 1, 3);
//
//                        _CAS(i + 0, j + 0, k + 0, 5);
//                        _CAS(i + 0, j + 1, k + 0, 5);
//
//                        _CAS(i + 0, j + 0, k + 0, 6);
//                        _CAS(i + 0, j + 0, k + 1, 6);
//                        break;
//                    case VOLUME:
//                        _CAS(i - 1, j - 1, k - 1, 7);
//                        _CAS(i + 0, j - 1, k - 1, 7);
//                        _CAS(i + 1, j - 1, k - 1, 7);
//                        _CAS(i - 1, j + 0, k - 1, 7);
//                        _CAS(i + 0, j + 0, k - 1, 7);
//                        _CAS(i + 1, j + 0, k - 1, 7);
//                        _CAS(i - 1, j + 1, k - 1, 7);
//                        _CAS(i + 0, j + 1, k - 1, 7);
//                        _CAS(i + 1, j + 1, k - 1, 7);
//
//                        _CAS(i - 1, j - 1, k + 0, 7);
//                        _CAS(i + 0, j - 1, k + 0, 7);
//                        _CAS(i + 1, j - 1, k + 0, 7);
//                        _CAS(i - 1, j + 0, k + 0, 7);
//                        //   _CAS(i + 0, j + 0, k + 0, 7);
//                        _CAS(i + 1, j + 0, k + 0, 7);
//                        _CAS(i - 1, j + 1, k + 0, 7);
//                        _CAS(i + 0, j + 1, k + 0, 7);
//                        _CAS(i + 1, j + 1, k + 0, 7);
//
//                        _CAS(i - 1, j - 1, k + 1, 7);
//                        _CAS(i + 0, j - 1, k + 1, 7);
//                        _CAS(i + 1, j - 1, k + 1, 7);
//                        _CAS(i - 1, j + 0, k + 1, 7);
//                        _CAS(i + 0, j + 0, k + 1, 7);
//                        _CAS(i + 1, j + 0, k + 1, 7);
//                        _CAS(i - 1, j + 1, k + 1, 7);
//                        _CAS(i + 0, j + 1, k + 1, 7);
//                        _CAS(i + 1, j + 1, k + 1, 7);
//                        break;
//                    default:
//                        break;
//                }
//#undef _CAS
//            }

    return m_interface_cache_.at(iform).at(tag_in).at(tag_out);
}
}
}  // namespace simpla{namespace get_mesh{