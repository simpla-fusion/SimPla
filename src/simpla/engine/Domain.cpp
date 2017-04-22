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
#include "Worker.h"
namespace simpla {
namespace engine {

struct Domain::pimpl_s {
    std::shared_ptr<Chart> m_chart_ = nullptr;
    std::shared_ptr<geometry::GeoObject> m_geo_obj_ = nullptr;
    std::shared_ptr<Worker> m_worker_ = nullptr;
    std::map<std::shared_ptr<geometry::GeoObject>, std::shared_ptr<Worker>> m_boundary_;

    std::shared_ptr<Patch> m_patch_;
};
Domain::Domain() : SPObject(), m_pimpl_(new pimpl_s) {}
Domain::~Domain() { Finalize(); }

void Domain::Initialize() {}
void Domain::SetUp() {
    m_pimpl_->m_worker_->GetMesh()->SetChart(m_pimpl_->m_chart_);
    m_pimpl_->m_worker_->SetUp();
}
void Domain::TearDown() {}
void Domain::Finalize() {}
void Domain::InitializeData(Real time_now) { m_pimpl_->m_worker_->InitializeCondition(time_now); }

std::shared_ptr<data::DataTable> Domain::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    res->SetValue("Type", "Domain");
    if (m_pimpl_->m_chart_ != nullptr) { res->Link("Chart", m_pimpl_->m_chart_->Serialize()); }
    if (m_pimpl_->m_geo_obj_ != nullptr) { res->Link("GeoObject", m_pimpl_->m_geo_obj_->Serialize()); }
    if (m_pimpl_->m_worker_ != nullptr) { res->Link("Worker", m_pimpl_->m_worker_->Serialize()); }
    return res;
}
void Domain::Deserialize(std::shared_ptr<data::DataTable> t) {
    m_pimpl_->m_chart_ = Chart::Create(t->GetTable("Chart"));
    m_pimpl_->m_geo_obj_ = geometry::GeoObject::Create(t->GetTable("GeoObject"));
    m_pimpl_->m_worker_ =
        Worker::Create(t->GetValue<std::string>("Worker/Type") + "<" + t->GetValue<std::string>("Chart/Type") + ">");
    // TODO: unfinished
}
void Domain::Register(AttributeGroup *attr_grp) { m_pimpl_->m_worker_->Register(attr_grp); }
void Domain::Deregister(AttributeGroup *attr_grp) { m_pimpl_->m_worker_->Deregister(attr_grp); }

void Domain::SetGeoObject(std::shared_ptr<geometry::GeoObject> g) { m_pimpl_->m_geo_obj_ = g; }
std::shared_ptr<geometry::GeoObject> Domain::GetGeoObject() const { return m_pimpl_->m_geo_obj_; }

void Domain::SetChart(std::shared_ptr<Chart> m) { m_pimpl_->m_chart_ = m; };
std::shared_ptr<Chart> Domain::GetChart() const { return m_pimpl_->m_chart_; }

void Domain::SetWorker(std::shared_ptr<Worker> w) { m_pimpl_->m_worker_ = w; }
std::shared_ptr<Worker> Domain::GetWorker() const { return m_pimpl_->m_worker_; }

void Domain::AddBoundaryCondition(std::shared_ptr<Worker> w, std::shared_ptr<geometry::GeoObject> g) {
    m_pimpl_->m_boundary_.emplace(g, w);
}
void Domain::Push(std::shared_ptr<Patch> p) { m_pimpl_->m_patch_ = p; }
std::shared_ptr<Patch> Domain::Pop() { return m_pimpl_->m_patch_; }

void Domain::InitializeCondition(Real time_now) {
    if (m_pimpl_->m_worker_ != nullptr && m_pimpl_->m_patch_ != nullptr) {
        m_pimpl_->m_worker_->Push(m_pimpl_->m_patch_);
        m_pimpl_->m_worker_->InitializeCondition(time_now);
        m_pimpl_->m_patch_ = m_pimpl_->m_worker_->Pop();
    }
}

void Domain::BoundaryCondition(Real time_now, Real time_dt) {
    if (m_pimpl_->m_worker_ != nullptr && m_pimpl_->m_patch_ != nullptr) {
        m_pimpl_->m_worker_->Push(m_pimpl_->m_patch_);
        m_pimpl_->m_worker_->BoundaryCondition(time_now, time_dt);
        m_pimpl_->m_patch_ = m_pimpl_->m_worker_->Pop();
    }
}

void Domain::Advance(Real time_now, Real time_dt) {
    if (m_pimpl_->m_worker_ != nullptr && m_pimpl_->m_patch_ != nullptr) {
        m_pimpl_->m_worker_->Push(m_pimpl_->m_patch_);
        m_pimpl_->m_worker_->Advance(time_now, time_dt);
        m_pimpl_->m_patch_ = m_pimpl_->m_worker_->Pop();
    }
}

//    auto mblk_ibx = p->GetBlock()->GetIndexBox();
//
//    auto x0 = GetChart()->GetOrigin();
//    auto dx = GetChart()->GetOrigin();
//    dx *= std::pow(0.5, p->GetBlock()->GetLevel());
//
//    box_type mblk_box;
//    std::get<0>(mblk_box) = x0 + std::get<0>(mblk_ibx) * dx;
//    std::get<1>(mblk_box) = x0 + std::get<1>(mblk_ibx) * dx;
//    CHECK(m_pimpl_->m_geo_obj_->GetBoundBox());
//    CHECK(mblk_box);
//
//    switch (m_pimpl_->m_geo_obj_->CheckOverlap(mblk_box)) {
//        case -1:
//            break;
//        case 0:
//            for (auto &item : m_pimpl_->m_boundary_) {
//                if (item.first == nullptr || item.first->CheckOverlap(mblk_box) < 1) {
//                    item.second->ConvertPatchFromSAMRAI(p);
//                    item.second->SetUp(time_now);
//                    item.second->ConvertPatchToSAMRAI(p);
//                }
//            }
//            break;
//        case 1:
//        default:
//            // DO NOTHING
//            break;
//    };

//    CHECK(p->GetBlock()->GetBoundBox());
//    if (p == nullptr) { return; }
//
//    auto mblk_ibx = p->GetBlock()->GetIndexBox();
//
//    auto x0 = GetChart()->GetOrigin();
//    auto dx = GetChart()->GetOrigin();
//    dx *= std::pow(0.5, p->GetBlock()->GetLevel());
//
//    box_type mblk_box;
//    std::get<0>(mblk_box) = x0 + std::get<0>(mblk_ibx) * dx;
//    std::get<1>(mblk_box) = x0 + std::get<1>(mblk_ibx) * dx;
//    CHECK(m_pimpl_->m_geo_obj_->GetBoundBox());
//    CHECK(mblk_box);
//
//    switch (m_pimpl_->m_geo_obj_->CheckOverlap(mblk_box)) {
//        case -1:

//            break;
//        case 0:
//            for (auto &item : m_pimpl_->m_boundary_) {
//                if (item.first == nullptr || item.first->CheckOverlap(mblk_box) < 1) {
//                    item.second->ConvertPatchFromSAMRAI(p);
//                    item.second->AdvanceData(time_now, time_dt);
//                    item.second->ConvertPatchToSAMRAI(p);
//                }
//            }
//            break;
//        case 1:
//        default:
//            // DO NOTHING
//            break;
//    };

// Domain *Domain::Clone() const { return new Domain(*this); }
//
// void Domain::ConvertPatchFromSAMRAI(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<data::DataTable> const &d) {
//    ASSERT(m != nullptr);
//    m_pimpl_->m_mesh_block_ = m;
//    if (m_pimpl_->m_patch_ == nullptr) { m_pimpl_->m_patch_ = std::make_shared<data::DataTable>(); }
//    ASSERT(d->isTable());
//    m_pimpl_->m_patch_->PushPatch(d->cast_as<data::DataTable>());
//};
// void Domain::ConvertPatchFromSAMRAI(std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataTable>> const &p)
// {
//    ConvertPatchFromSAMRAI(p.first, p.second);
//}
// std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataTable>> Domain::ConvertPatchToSAMRAI() {
//    auto res =
//        std::make_pair(m_pimpl_->m_chart_->GetBlock(),
//        std::dynamic_pointer_cast<data::DataTable>(m_pimpl_->m_patch_));
//
//    m_pimpl_->m_patch_.reset();
//    return (res);
//};
//
// void Domain::Run(Real dt) {
//    //    m_pimpl_->m_chart_->ConvertPatchFromSAMRAI(m_pimpl_->m_mesh_block_, m_pimpl_->m_patch_);
//    //
//    //    for (auto &item : m_pimpl_->m_worker_) {
//    //        ASSERT(m_pimpl_->m_chart_ != nullptr);
//    //        item.second->SetMesh(m_pimpl_->m_chart_.get());
//    //        item.second->ConvertPatchFromSAMRAI(m_pimpl_->m_chart_->GetBlock(), m_pimpl_->m_patch_);
//    //        item.second->Run(dt);
//    //        auto res = item.second->PopPatch();
//    //
//    //        ConvertPatchFromSAMRAI(res);  // item.second->PopPatch());
//    //    }
//}
// void Domain::Attach(AttributeGroup *p) {
//    if (p == nullptr) { return; }
//    auto res = m_pimpl_->m_attr_bundle_.emplace(p);
//
//    if (res.second) {
//        (*res.first)->Connect(this);
//        for (Attribute *v : (*res.first)->GetAllAttributes()) {
//            ASSERT(v != nullptr)
//            if (v->name() != "") {
//                auto t = db()->PopPatch("Attributes/" + v->name());
//                if (t == nullptr || !t->isTable()) {
//                    t = v->db();
//                } else {
//                    t->cast_as<data::DataTable>().PushPatch(*v->db());
//                }
//                db()->Link("Attributes/" + v->name(), t);
//            }
//        };
//    }
//}
// void Domain::Detach(AttributeGroup *p) {
//    if (p != nullptr && m_pimpl_->m_attr_bundle_.erase(p) > 0) {}
//}
//
// std::set<Attribute *> const &Domain::GetAllAttributes() const { return m_pimpl_->m_attributes_; }
//
// void Domain::InitializeConditionPatch() {
//    auto m = m_pimpl_->m_chart_->CreateView(nullptr, m_pimpl_->m_geo_obj_);
//    m_pimpl_->m_worker_->SetMesh(m);
//    for (auto *v : m_pimpl_->m_attr_bundle_.GetAll()) { v->SetMesh(m.get()); }
//
//    if (m_pimpl_->m_chart_ != nullptr) { return; }
//    m_pimpl_->m_chart_.reset(Mesh::Create(db()->GetTable("Mesh")));
//    ASSERT(m_pimpl_->m_chart_ != nullptr);
//    db()->Link("Mesh", m_pimpl_->m_chart_->db());
//    auto t_worker = db()->PopPatch("Task");
//
//    if (t_worker != nullptr && t_worker->isArray()) {
//        t_worker->cast_as<data::DataArray>().Foreach([&](std::shared_ptr<data::DataEntity> const &c) {
//            std::shared_ptr<Task> res(
//                Task::Create(std::dynamic_pointer_cast<data::DataTable>(c)->GetValue<std::string>("name", "")));
//            SetWorker(res);
//        });
//    }
//    for (auto const &attr_bundle : m_pimpl_->m_attr_bundle_) {
//        for (auto *v : attr_bundle->GetAllAttributes()) { m_pimpl_->m_attributes_.insert(v); }
//    }
//    LOGGER << "Domain View [" << name() << "] is initialized!" << std::endl;
//}
//
// std::pair<std::shared_ptr<Task>, bool> Domain::SetWorker(std::shared_ptr<Task> const &w, int pos) {
//    ASSERT(w != nullptr);
//    auto res = m_pimpl_->m_worker_.emplace(pos, w);
//    if (res.second) { Attach(res.first->second.get()); }
//    return std::make_pair(res.first->second, res.second);
//}
//
// void Domain::RemoveWorker(std::shared_ptr<Task> const &w) {
//    UNIMPLEMENTED;
//    //    auto it = m_backend_->m_worker_.find(w);
//    //    if (it != m_backend_->m_worker_.end()) { m_backend_->m_worker_.Disconnect(it); }
//};
//
// std::ostream &Domain::Print(std::ostream &os, int indent) const {
//    if (m_pimpl_->m_chart_ != nullptr) {
//        os << " Mesh = { ";
//        m_pimpl_->m_chart_->Print(os, indent);
//        os << " }, " << std::endl;
//    }
//
//    if (m_pimpl_->m_worker_.size() > 0) {
//        os << " Task = { ";
//        for (auto &item : m_pimpl_->m_worker_) { item.second->Print(os, indent); }
//        os << " } " << std::endl;
//    }
//    return os;
//};
//
// void Model::UpdatePatch(Real data_time, Real dt) {
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
//    //                auto x = m_chart_->mesh_block()->point(i, j, k);
//    //                auto& GetTag = m_tags_(i, j, k, 0);
//    //
//    //                GetTag = VACUUM;
//    //
//    //                for (auto const& obj : m_g_objs_) {
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
//    m_range_cache_.Disconnect(m_chart_->mesh_block()->id());
//    m_interface_cache_.Disconnect(m_chart_->mesh_block()->id());
//    PostProcess();
//};
//
// Range<id_type> const& Model::select(int GetIFORM, std::string const& GetTag) { return select(GetIFORM,
// m_g_name_map_.at(GetTag));
// }
//
// Range<id_type> const& Model::select(int GetIFORM, int GetTag) {
//    //    typedef EntityIdCoder M;
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
//    //    typedef EntityIdCoder M;
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