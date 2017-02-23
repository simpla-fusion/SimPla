//
// Created by salmon on 17-2-12.
//
#include "DomainView.h"
#include <simpla/SIMPLA_config.h>
#include <simpla/concept/StateCounter.h>
#include <set>
#include "AttributeView.h"
#include "MeshBlock.h"
#include "MeshView.h"
#include "Object.h"
#include "Patch.h"
#include "Worker.h"

namespace simpla {
namespace engine {
struct DomainView::pimpl_s {
    id_type m_current_block_id_ = NULL_ID;
    std::shared_ptr<MeshView> m_mesh_;
    std::list<std::shared_ptr<Worker>> m_workers_;
    std::shared_ptr<Patch> m_patch_;
    std::map<id_type, std::shared_ptr<AttributeDesc>> m_attrs_dict_;
    Manager *m_manager_ = nullptr;

    //    DataAttribute<int, VERTEX, 9> m_tags_{"tags", "INPUT"_};
    //    std::map<int, std::map<int, Range<entity_id>>> m_range_cache_;
    //    std::map<int, std::map<int, std::map<int, Range<entity_id>>>> m_interface_cache_;
};

DomainView::DomainView() : m_pimpl_(new pimpl_s) {}
DomainView::~DomainView() {}

Manager const *DomainView::GetManager(Manager *) const { return m_pimpl_->m_manager_; }
void DomainView::SetManager(Manager *m) {
    concept::StateCounter::Click();
    m_pimpl_->m_manager_ = m;
}

/**
 *
 * @startuml
 * actor Main
 * Main -> DomainView : Set U as MeshView
 * activate DomainView
 *     alt if MeshView=nullptr
 *          create MeshView
 *     DomainView -> MeshView : create U as MeshView
 *     MeshView --> DomainView: return MeshView
 *     end
 *     DomainView --> Main : Done
 * deactivate DomainView
 * @enduml
 * @startuml
 * actor Main
 * Main -> DomainView : Dispatch
 * activate DomainView
 *     DomainView->MeshView:  Dispatch
 *     MeshView->MeshView: SetMeshBlock
 *     activate MeshView
 *     deactivate MeshView
 *     MeshView -->DomainView:  Done
*      DomainView --> Main : Done
 * deactivate DomainView
 * @enduml
 * @startuml
 * Main ->DomainView: Update
 * activate DomainView
 *     DomainView -> AttributeView : Update
 *     activate AttributeView
 *          AttributeView -> Field : Update
 *          Field -> AttributeView : Update
 *          activate AttributeView
 *               AttributeView -> DomainView : get DataBlock at attr.id()
 *               DomainView --> AttributeView : return DataBlock at attr.id()
 *               AttributeView --> Field : return DataBlock is ready
 *          deactivate AttributeView
 *          alt if data_block.isNull()
 *              Field -> Field :  create DataBlock
 *              Field -> AttributeView : send DataBlock
 *              AttributeView --> Field : Done
 *          end
 *          Field --> AttributeView : Done
 *          AttributeView --> DomainView : Done
 *     deactivate AttributeView
 *     DomainView -> MeshView : Update
 *     activate MeshView
 *          alt if isFirstTime
 *              MeshView -> AttributeView : Set Initialize Value
 *              activate AttributeView
 *                   AttributeView --> MeshView : Done
 *              deactivate AttributeView
 *          end
 *          MeshView --> DomainView : Done
 *     deactivate MeshView
 *     DomainView -> Worker : Update
 *     activate Worker
 *          alt if isFirstTime
 *              Worker -> AttributeView : set initialize value
 *              activate AttributeView
 *                  AttributeView --> Worker : Done
 *              deactivate AttributeView
 *          end
 *          Worker --> DomainView : Done
 *     deactivate Worker
 *     DomainView --> Main : Done
 * deactivate DomainView
 * deactivate Main
 * @enduml
 */
void DomainView::Dispatch(std::shared_ptr<Patch> p) { m_pimpl_->m_patch_ = p; };
std::map<id_type, std::shared_ptr<engine::AttributeDesc>> const &DomainView::GetAttributeDict() const {
    return m_pimpl_->m_attrs_dict_;
};
data::DataTable const &DomainView::attr_db(id_type id) const { return (m_pimpl_->m_attrs_dict_.at(id)->db()); };
data::DataTable &DomainView::attr_db(id_type id) { return (m_pimpl_->m_attrs_dict_.at(id)->db()); };
id_type DomainView::current_block_id() const { return m_pimpl_->m_current_block_id_; }

bool DomainView::isUpdated() const {
    return (!concept::StateCounter::isModified()) && (m_pimpl_->m_current_block_id_ == GetMeshBlockId());
}

void DomainView::Update() {
    if (isUpdated()) { return; }

    if (m_pimpl_->m_patch_ == nullptr) { m_pimpl_->m_patch_ = std::make_shared<Patch>(); }
    if (m_pimpl_->m_mesh_ != nullptr) { m_pimpl_->m_mesh_->Update(); }
    for (auto &item : m_pimpl_->m_workers_) { item->Update(); }
    m_pimpl_->m_current_block_id_ = m_pimpl_->m_patch_->GetMeshBlock()->id();

    concept::StateCounter::Recount();
}

void DomainView::Evaluate() {
    for (auto &item : m_pimpl_->m_workers_) { item->Evaluate(); }
}

void DomainView::SetMesh(std::shared_ptr<MeshView> const &m) {
    concept::StateCounter::Click();
    m_pimpl_->m_mesh_ = m;
    m_pimpl_->m_mesh_->SetDomain(this);
};

const MeshView * DomainView::GetMesh() const { return m_pimpl_->m_mesh_.get(); }
void DomainView::AppendWorker(std::shared_ptr<Worker> const &w) {
    concept::StateCounter::Click();
    ASSERT(w != nullptr);
    w->SetDomain(this);
    m_pimpl_->m_workers_.push_back(w);
};

void DomainView::PrependWorker(std::shared_ptr<Worker> const &w) {
    concept::StateCounter::Click();
    ASSERT(w != nullptr);
    w->SetDomain(this);
    m_pimpl_->m_workers_.push_front(w);
};

void DomainView::RemoveWorker(std::shared_ptr<Worker> const &w) {
    concept::StateCounter::Click();
    UNIMPLEMENTED;
    //    auto it = m_pimpl_->m_workers_.find(w);
    //    if (it != m_pimpl_->m_workers_.end()) { m_pimpl_->m_workers_.erase(it); }
};
id_type DomainView::GetMeshBlockId() const { return GetMeshBlock() == nullptr ? NULL_ID : GetMeshBlock()->id(); }
std::shared_ptr<MeshBlock> const &DomainView::GetMeshBlock() const { return m_pimpl_->m_patch_->GetMeshBlock(); };
std::shared_ptr<DataBlock> const &DomainView::GetDataBlock(id_type id) const {
    return m_pimpl_->m_patch_->GetDataBlock(id);
}
std::shared_ptr<DataBlock> &DomainView::GetDataBlock(id_type id) { return m_pimpl_->m_patch_->GetDataBlock(id); }

void DomainView::SetDataBlock(id_type id, std::shared_ptr<DataBlock> const &d) {
    m_pimpl_->m_patch_->SetDataBlock(id, d);
}

std::ostream &DomainView::Print(std::ostream &os, int indent) const {
    if (m_pimpl_->m_mesh_ != nullptr) {
        os << " Mesh = { ";
        m_pimpl_->m_mesh_->Print(os, indent);
        os << " }, " << std::endl;
    }

    if (m_pimpl_->m_workers_.size() > 0) {
        os << " Worker = { ";
        for (auto &item : m_pimpl_->m_workers_) { item->Print(os, indent); }
        os << " } " << std::endl;
    }

    return os;
};
//
// void Model::Update(Real data_time, Real dt) {
//    PreProcess();
//    //
//    //    index_type const* lower = m_tags_.lower();
//    //    index_type const* upper = m_tags_.upper();
//    //
//    //    index_type ib = lower[0];
//    //    index_type ie = upper[0];
//    //    index_type jb = lower[1];
//    //    index_type je = upper[1];
//    //    index_type kb = lower[2];
//    //    index_type ke = upper[2];
//    //
//    //    for (index_type i = ib; i < ie; ++i)
//    //        for (index_type j = jb; j < je; ++j)
//    //            for (index_type k = kb; k < ke; ++k) {
//    //                auto x = m_mesh_->mesh_block()->point(i, j, k);
//    //                auto& tag = m_tags_(i, j, k, 0);
//    //
//    //                tag = VACUUM;
//    //
//    //                for (auto const& obj : m_g_obj_) {
//    //                    if (obj.second->check_inside(x)) { tag |= obj.first; }
//    //                }
//    //            }
//    //    for (index_type i = ib; i < ie - 1; ++i)
//    //        for (index_type j = jb; j < je - 1; ++j)
//    //            for (index_type k = kb; k < ke - 1; ++k) {
//    //                m_tags_(i, j, k, 1) = m_tags_(i, j, k, 0) | m_tags_(i + 1, j, k, 0);
//    //                m_tags_(i, j, k, 2) = m_tags_(i, j, k, 0) | m_tags_(i, j + 1, k, 0);
//    //                m_tags_(i, j, k, 4) = m_tags_(i, j, k, 0) | m_tags_(i, j, k + 1, 0);
//    //            }
//    //
//    //    for (index_type i = ib; i < ie - 1; ++i)
//    //        for (index_type j = jb; j < je - 1; ++j)
//    //            for (index_type k = kb; k < ke - 1; ++k) {
//    //                m_tags_(i, j, k, 3) = m_tags_(i, j, k, 1) | m_tags_(i, j + 1, k, 1);
//    //                m_tags_(i, j, k, 5) = m_tags_(i, j, k, 1) | m_tags_(i, j, k + 1, 1);
//    //                m_tags_(i, j, k, 6) = m_tags_(i, j + 1, k, 1) | m_tags_(i, j, k + 1, 1);
//    //            }
//    //
//    //    for (index_type i = ib; i < ie - 1; ++i)
//    //        for (index_type j = jb; j < je - 1; ++j)
//    //            for (index_type k = kb; k < ke - 1; ++k) {
//    //                m_tags_(i, j, k, 7) = m_tags_(i, j, k, 3) | m_tags_(i, j, k + 1, 3);
//    //            }
//};
//
// void Model::Finalize(Real data_time, Real dt) {
//    m_range_cache_.erase(m_mesh_->mesh_block()->id());
//    m_interface_cache_.erase(m_mesh_->mesh_block()->id());
//    PostProcess();
//};
//
// Range<id_type> const& Model::select(int iform, std::string const& tag) { return select(iform, m_g_name_map_.at(tag));
// }
//
// Range<id_type> const& Model::select(int iform, int tag) {
//    //    typedef MeshEntityIdCoder M;
//    //
//    //    try {
//    //        return m_range_cache_.at(iform).at(tag);
//    //    } catch (...) {}
//    //
//    //    const_cast<this_type*>(this)->m_range_cache_[iform].emplace(
//    //        std::make_pair(tag, Range<id_type>(std::make_shared<UnorderedRange<id_type>>())));
//    //
//    //    auto& res = *m_range_cache_.at(iform).at(tag).self().template as<UnorderedRange<id_type>>();
//    //
//    //    index_type const* lower = m_tags_.lower();
//    //    index_type const* upper = m_tags_.upper();
//    //
//    //    index_type ib = lower[0];
//    //    index_type ie = upper[0];
//    //    index_type jb = lower[1];
//    //    index_type je = upper[1];
//    //    index_type kb = lower[2];
//    //    index_type ke = upper[2];
//    //
//    //#define _CAS(I, J, K, L) \
////    if (I >= 0 && J >= 0 && K >= 0 && ((m_tags_(I, J, K, L) & tag) == tag)) { res.insert(M::pack_index(I, J, K, L)); }
//    //
//    //    switch (iform) {
//    //        case VERTEX:
//    //#pragma omp parallel for
//    //            for (index_type i = ib; i < ie; ++i)
//    //                for (index_type j = jb; j < je; ++j)
//    //                    for (index_type k = kb; k < ke; ++k) { _CAS(i, j, k, 0); }
//    //
//    //            break;
//    //        case EDGE:
//    //#pragma omp parallel for
//    //            for (index_type i = ib; i < ie - 1; ++i)
//    //                for (index_type j = jb; j < je - 1; ++j)
//    //                    for (index_type k = kb; k < ke - 1; ++k) {
//    //                        _CAS(i, j, k, 1);
//    //                        _CAS(i, j, k, 2);
//    //                        _CAS(i, j, k, 4);
//    //                    }
//    //            break;
//    //        case FACE:
//    //#pragma omp parallel for
//    //            for (index_type i = ib; i < ie - 1; ++i)
//    //                for (index_type j = jb; j < je - 1; ++j)
//    //                    for (index_type k = kb; k < ke - 1; ++k) {
//    //                        _CAS(i, j, k, 3);
//    //                        _CAS(i, j, k, 5);
//    //                        _CAS(i, j, k, 6);
//    //                    }
//    //            break;
//    //        case VOLUME:
//    //#pragma omp parallel for
//    //            for (index_type i = ib; i < ie - 1; ++i)
//    //                for (index_type j = jb; j < je - 1; ++j)
//    //                    for (index_type k = kb; k < ke - 1; ++k) { _CAS(i, j, k, 7); }
//    //            break;
//    //        default:
//    //            break;
//    //    }
//    //#undef _CAS
//    //    return m_range_cache_.at(iform).at(tag);
//    //    ;
//}
//
///**
// *  id < 0 out of surface
// *       = 0 on surface
// *       > 0 in surface
// */
// Range<id_type> const& Model::interface(int iform, const std::string& s_in, const std::string& s_out) {
//    return interface(iform, m_g_name_map_.at(s_in), m_g_name_map_.at(s_out));
//}
//
// Range<id_type> const& Model::interface(int iform, int tag_in, int tag_out) {
//    //    try {
//    //        return m_interface_cache_.at(iform).at(tag_in).at(tag_out);
//    //    } catch (...) {}
//    //
//    //    typedef mesh::MeshEntityIdCoder M;
//    //
//    //    const_cast<this_type*>(this)->m_interface_cache_[iform][tag_in].emplace(
//    //        std::make_pair(tag_out, Range<id_type>(std::make_shared<UnorderedRange<id_type>>())));
//    //
//    //    auto& res = *const_cast<this_type*>(this)
//    //                     ->m_interface_cache_.at(iform)
//    //                     .at(tag_in)
//    //                     .at(tag_out)
//    //                     .self()
//    //                     .template as<UnorderedRange<id_type>>();
//    //
//    //    index_type const* lower = m_tags_.lower();
//    //    index_type const* upper = m_tags_.upper();
//    //
//    //    index_type ib = lower[0];
//    //    index_type ie = upper[0];
//    //    index_type jb = lower[1];
//    //    index_type je = upper[1];
//    //    index_type kb = lower[2];
//    //    index_type ke = upper[2];
//    //
//    //    int v_tag = tag_in | tag_out;
//    //#pragma omp parallel for
//    //    for (index_type i = ib; i < ie - 1; ++i)
//    //        for (index_type j = jb; j < je - 1; ++j)
//    //            for (index_type k = kb; k < ke - 1; ++k) {
//    //                if ((m_tags_(i, j, k, 7) & v_tag) != v_tag) { continue; }
//    //#define _CAS(I, J, K, L) \
////    if (I >= 0 && J >= 0 && K >= 0 && m_tags_(I, J, K, L) == tag_in) { res.insert(M::pack_index(I, J, K, L)); }
//    //                switch (iform) {
//    //                    case VERTEX:
//    //                        _CAS(i + 0, j + 0, k + 0, 0);
//    //                        _CAS(i + 1, j + 0, k + 0, 0);
//    //                        _CAS(i + 0, j + 1, k + 0, 0);
//    //                        _CAS(i + 1, j + 1, k + 0, 0);
//    //                        _CAS(i + 0, j + 0, k + 1, 0);
//    //                        _CAS(i + 1, j + 0, k + 1, 0);
//    //                        _CAS(i + 0, j + 1, k + 1, 0);
//    //                        _CAS(i + 1, j + 1, k + 1, 0);
//    //
//    //                        break;
//    //                    case EDGE:
//    //                        _CAS(i + 0, j + 0, k + 0, 1);
//    //                        _CAS(i + 0, j + 1, k + 0, 1);
//    //                        _CAS(i + 0, j + 0, k + 1, 1);
//    //                        _CAS(i + 0, j + 1, k + 1, 1);
//    //
//    //                        _CAS(i + 0, j + 0, k + 0, 2);
//    //                        _CAS(i + 1, j + 0, k + 0, 2);
//    //                        _CAS(i + 0, j + 0, k + 1, 2);
//    //                        _CAS(i + 1, j + 0, k + 1, 2);
//    //
//    //                        _CAS(i + 0, j + 0, k + 0, 4);
//    //                        _CAS(i + 0, j + 1, k + 0, 4);
//    //                        _CAS(i + 1, j + 0, k + 0, 4);
//    //                        _CAS(i + 1, j + 1, k + 0, 4);
//    //                        break;
//    //                    case FACE:
//    //                        _CAS(i + 0, j + 0, k + 0, 3);
//    //                        _CAS(i + 0, j + 0, k + 1, 3);
//    //
//    //                        _CAS(i + 0, j + 0, k + 0, 5);
//    //                        _CAS(i + 0, j + 1, k + 0, 5);
//    //
//    //                        _CAS(i + 0, j + 0, k + 0, 6);
//    //                        _CAS(i + 0, j + 0, k + 1, 6);
//    //                        break;
//    //                    case VOLUME:
//    //                        _CAS(i - 1, j - 1, k - 1, 7);
//    //                        _CAS(i + 0, j - 1, k - 1, 7);
//    //                        _CAS(i + 1, j - 1, k - 1, 7);
//    //                        _CAS(i - 1, j + 0, k - 1, 7);
//    //                        _CAS(i + 0, j + 0, k - 1, 7);
//    //                        _CAS(i + 1, j + 0, k - 1, 7);
//    //                        _CAS(i - 1, j + 1, k - 1, 7);
//    //                        _CAS(i + 0, j + 1, k - 1, 7);
//    //                        _CAS(i + 1, j + 1, k - 1, 7);
//    //
//    //                        _CAS(i - 1, j - 1, k + 0, 7);
//    //                        _CAS(i + 0, j - 1, k + 0, 7);
//    //                        _CAS(i + 1, j - 1, k + 0, 7);
//    //                        _CAS(i - 1, j + 0, k + 0, 7);
//    //                        //   _CAS(i + 0, j + 0, k + 0, 7);
//    //                        _CAS(i + 1, j + 0, k + 0, 7);
//    //                        _CAS(i - 1, j + 1, k + 0, 7);
//    //                        _CAS(i + 0, j + 1, k + 0, 7);
//    //                        _CAS(i + 1, j + 1, k + 0, 7);
//    //
//    //                        _CAS(i - 1, j - 1, k + 1, 7);
//    //                        _CAS(i + 0, j - 1, k + 1, 7);
//    //                        _CAS(i + 1, j - 1, k + 1, 7);
//    //                        _CAS(i - 1, j + 0, k + 1, 7);
//    //                        _CAS(i + 0, j + 0, k + 1, 7);
//    //                        _CAS(i + 1, j + 0, k + 1, 7);
//    //                        _CAS(i - 1, j + 1, k + 1, 7);
//    //                        _CAS(i + 0, j + 1, k + 1, 7);
//    //                        _CAS(i + 1, j + 1, k + 1, 7);
//    //                        break;
//    //                    default:
//    //                        break;
//    //                }
//    //#undef _CAS
//    //            }
//
//    return m_interface_cache_.at(iform).at(tag_in).at(tag_out);
//}
}  // namespace engine
}  // namespace simpla