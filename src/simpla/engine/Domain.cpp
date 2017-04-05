//
// Created by salmon on 17-4-1.
//
#include "Domain.h"
#include <simpla/SIMPLA_config.h>
#include <set>
#include "Attribute.h"
#include "Mesh.h"
#include "MeshBlock.h"
#include "Patch.h"
#include "SPObject.h"
#include "Task.h"
namespace simpla {
namespace engine {

struct Domain::pimpl_s {
    std::shared_ptr<geometry::GeoObject> m_geo_obj_;
    std::shared_ptr<Mesh> m_mesh_;
    std::map<int, std::shared_ptr<Task>> m_workers_;
    std::set<AttributeBundle *> m_attr_bundle_;
    std::set<Attribute *> m_attributes_;

    std::shared_ptr<MeshBlock> m_mesh_block_;
    std::shared_ptr<data::DataTable> m_patch_;
};

Domain::Domain(std::shared_ptr<data::DataTable> const &t, std::shared_ptr<geometry::GeoObject> const &g)
    : concept::Configurable(t), m_pimpl_(new pimpl_s) {
    m_pimpl_->m_geo_obj_ = g;
}

Domain::Domain(std::shared_ptr<Mesh> const &m) : m_pimpl_(new pimpl_s) {
    if (m != nullptr) { m_pimpl_->m_geo_obj_ = m->GetGeoObject(); }
}

Domain::Domain(const Domain &other) : m_pimpl_(new pimpl_s) { UNIMPLEMENTED; }

Domain::~Domain() {
    for (auto *item : m_pimpl_->m_attr_bundle_) { Detach(item); }
    Finalize();
}

// void Domain::RegisterMeshFactory(
//    std::function<std::shared_ptr<Mesh>(std::shared_ptr<data::DataTable> const &,
//                                            std::shared_ptr<geometry::GeoObject> const &)> const &f) {
//    m_pimpl_->m_mesh_factory_ = f;
//};

std::shared_ptr<geometry::GeoObject> const &Domain::GetGeoObject() const { return m_pimpl_->m_geo_obj_; }

std::shared_ptr<Mesh> const &Domain::GetMeshView() const { return m_pimpl_->m_mesh_; }

void Domain::SetMeshView(std::shared_ptr<Mesh> const &m) {
    m_pimpl_->m_mesh_ = m;
    db()->Set("Mesh", m->db());
    m_pimpl_->m_geo_obj_ = m->GetGeoObject();
};

std::shared_ptr<Domain> Domain::Clone() const { return std::make_shared<Domain>(*this); }

// std::shared_ptr<Mesh> Domain::CreateMeshView() {
//    std::shared_ptr<Mesh> m = nullptr;
//    if (m_pimpl_->m_mesh_ != nullptr) {
//        m = m_pimpl_->m_mesh_->Duplicate();
//    } else if (m_pimpl_->m_mesh_factory_) {
//        m = m_pimpl_->m_mesh_factory_(db()->GetTable("Mesh"), GetGeoObject());
//    }
//};
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
void Domain::PushData(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<data::DataTable> const &d) {
    ASSERT(m != nullptr);
    m_pimpl_->m_mesh_block_ = m;
    if (m_pimpl_->m_patch_ == nullptr) { m_pimpl_->m_patch_ = std::make_shared<data::DataTable>(); }
    ASSERT(d->isTable());
    m_pimpl_->m_patch_->Set(d->cast_as<data::DataTable>());
};
void Domain::PushData(std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataTable>> const &p) {
    PushData(p.first, p.second);
}
std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataTable>> Domain::PopData() {
    auto res = std::make_pair(m_pimpl_->m_mesh_->GetMeshBlock(),
                              std::dynamic_pointer_cast<data::DataTable>(m_pimpl_->m_patch_));

    m_pimpl_->m_patch_.reset();
    return (res);
};

void Domain::Run(Real dt) {
    m_pimpl_->m_mesh_->PushData(m_pimpl_->m_mesh_block_, m_pimpl_->m_patch_);

    for (auto &item : m_pimpl_->m_workers_) {
        ASSERT(m_pimpl_->m_mesh_ != nullptr);
        item.second->SetMesh(m_pimpl_->m_mesh_.get());
        item.second->PushData(m_pimpl_->m_mesh_->GetMeshBlock(), m_pimpl_->m_patch_);
        item.second->Run(dt);
        auto res = item.second->PopData();

        PushData(res);  // item.second->PopData());
    }
}
void Domain::Attach(AttributeBundle *p) {
    if (p == nullptr) { return; }
    auto res = m_pimpl_->m_attr_bundle_.emplace(p);

    if (res.second) {
        (*res.first)->Connect(this);
        for (Attribute *v : (*res.first)->GetAllAttributes()) {
            ASSERT(v != nullptr)
            if (v->name() != "") {
                auto t = db()->Get("Attributes/" + v->name());
                if (t == nullptr || !t->isTable()) {
                    t = v->db();
                } else {
                    t->cast_as<data::DataTable>().Set(*v->db());
                }
                db()->Link("Attributes/" + v->name(), t);
            }
        };
    }
}
void Domain::Detach(AttributeBundle *p) {
    if (p != nullptr && m_pimpl_->m_attr_bundle_.erase(p) > 0) {}
}

std::set<Attribute *> const &Domain::GetAllAttributes() const { return m_pimpl_->m_attributes_; }

void Domain::Initialize() {
    if (m_pimpl_->m_mesh_ != nullptr) { return; }
    m_pimpl_->m_mesh_ = GLOBAL_MESHVIEW_FACTORY.Create(db()->GetTable("Mesh"), m_pimpl_->m_geo_obj_);
    ASSERT(m_pimpl_->m_mesh_ != nullptr);
    db()->Link("Mesh", m_pimpl_->m_mesh_->db());
    auto t_worker = db()->Get("Task");

    if (t_worker != nullptr && t_worker->isArray()) {
        t_worker->cast_as<data::DataArray>().Foreach([&](std::shared_ptr<data::DataEntity> const &c) {
            auto res = GLOBAL_WORKER_FACTORY.Create(m_pimpl_->m_mesh_, c);
            AddWorker(res);
        });
    }
    for (auto const &attr_bundle : m_pimpl_->m_attr_bundle_) {
        for (auto *v : attr_bundle->GetAllAttributes()) { m_pimpl_->m_attributes_.insert(v); }
    }
    LOGGER << "Domain View [" << name() << "] is initialized!" << std::endl;
}
void Domain::Finalize() { m_pimpl_.reset(new pimpl_s); }

std::pair<std::shared_ptr<Task>, bool> Domain::AddWorker(std::shared_ptr<Task> const &w, int pos) {
    ASSERT(w != nullptr);
    auto res = m_pimpl_->m_workers_.emplace(pos, w);
    if (res.second) { Attach(res.first->second.get()); }
    return std::make_pair(res.first->second, res.second);
}

void Domain::RemoveWorker(std::shared_ptr<Task> const &w) {
    UNIMPLEMENTED;
    //    auto it = m_backend_->m_workers_.find(w);
    //    if (it != m_backend_->m_workers_.end()) { m_backend_->m_workers_.Disconnect(it); }
};

// std::ostream &Domain::Print(std::ostream &os, int indent) const {
//    if (m_pimpl_->m_mesh_ != nullptr) {
//        os << " Mesh = { ";
//        m_pimpl_->m_mesh_->Print(os, indent);
//        os << " }, " << std::endl;
//    }
//
//    if (m_pimpl_->m_workers_.size() > 0) {
//        os << " Task = { ";
//        for (auto &item : m_pimpl_->m_workers_) { item.second->Print(os, indent); }
//        os << " } " << std::endl;
//    }
//    return os;
//};
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
//    //                auto& GetTag = m_tags_(i, j, k, 0);
//    //
//    //                GetTag = VACUUM;
//    //
//    //                for (auto const& obj : m_g_obj_) {
//    //                    if (obj.second->check_inside(x)) { GetTag |= obj.first; }
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
//    m_range_cache_.Disconnect(m_mesh_->mesh_block()->id());
//    m_interface_cache_.Disconnect(m_mesh_->mesh_block()->id());
//    PostProcess();
//};
//
// Range<id_type> const& Model::select(int GetIFORM, std::string const& GetTag) { return select(GetIFORM,
// m_g_name_map_.at(GetTag));
// }
//
// Range<id_type> const& Model::select(int GetIFORM, int GetTag) {
//    //    typedef MeshEntityIdCoder M;
//    //
//    //    try {
//    //        return m_range_cache_.at(GetIFORM).at(GetTag);
//    //    } catch (...) {}
//    //
//    //    const_cast<this_type*>(this)->m_range_cache_[GetIFORM].emplace(
//    //        std::make_pair(GetTag, Range<id_type>(std::make_shared<UnorderedRange<id_type>>())));
//    //
//    //    auto& res = *m_range_cache_.at(GetIFORM).at(GetTag).self().template as<UnorderedRange<id_type>>();
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
////    if (I >= 0 && J >= 0 && K >= 0 && ((m_tags_(I, J, K, L) & GetTag) == GetTag)) { res.Connect(M::pack_index(I, J, K, L)); }
//    //
//    //    switch (GetIFORM) {
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
//    //    return m_range_cache_.at(GetIFORM).at(GetTag);
//    //    ;
//}
//
///**
// *  id < 0 out of surface
// *       = 0 on surface
// *       > 0 in surface
// */
// Range<id_type> const& Model::interface(int GetIFORM, const std::string& s_in, const std::string& s_out) {
//    return interface(GetIFORM, m_g_name_map_.at(s_in), m_g_name_map_.at(s_out));
//}
//
// Range<id_type> const& Model::interface(int GetIFORM, int tag_in, int tag_out) {
//    //    try {
//    //        return m_interface_cache_.at(GetIFORM).at(tag_in).at(tag_out);
//    //    } catch (...) {}
//    //
//    //    typedef mesh::MeshEntityIdCoder M;
//    //
//    //    const_cast<this_type*>(this)->m_interface_cache_[GetIFORM][tag_in].emplace(
//    //        std::make_pair(tag_out, Range<id_type>(std::make_shared<UnorderedRange<id_type>>())));
//    //
//    //    auto& res = *const_cast<this_type*>(this)
//    //                     ->m_interface_cache_.at(GetIFORM)
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
////    if (I >= 0 && J >= 0 && K >= 0 && m_tags_(I, J, K, L) == tag_in) { res.Connect(M::pack_index(I, J, K, L)); }
//    //                switch (GetIFORM) {
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
//    return m_interface_cache_.at(GetIFORM).at(tag_in).at(tag_out);
//}

}  // namespace engine {
}  // namespace simpla {