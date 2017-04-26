//
// Created by salmon on 17-2-16.
//
#include "Context.h"
#include <simpla/algebra/all.h>
#include <simpla/algebra/nTupleExt.h>
#include <simpla/data/all.h>
#include <simpla/geometry/GeoAlgorithm.h>
#include "Chart.h"
#include "Domain.h"
#include "MeshBase.h"
namespace simpla {
namespace engine {

struct Context::pimpl_s {
    std::map<std::string, std::shared_ptr<Attribute>> m_global_attributes_;
    std::map<std::string, std::shared_ptr<Domain>> m_domains_;
    Model m_model_;
    Atlas m_atlas_;
};

Context::Context() : m_pimpl_(new pimpl_s) {}
Context::~Context() {}

std::shared_ptr<data::DataTable> Context::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    for (auto const &item : m_pimpl_->m_domains_) { res->Link("Domain/" + item.first, item.second->Serialize()); }
    return res;
}
void Context::Deserialize(const std::shared_ptr<DataTable> &cfg) {
    //    auto worker_t = cfg->GetTable("Domains");
    //    if (worker_t == nullptr) { return; }
    //    worker_t->Foreach([&](std::string const &k, std::shared_ptr<DataEntity> t) {
    //        auto v = std::make_shared<Domain>();
    //        if (t->isTable()) { v->Deserialize(std::dynamic_pointer_cast<data::DataTable>(t)); }
    //        SetDomain(k, v);
    //    });
}

void Context::Initialize() {}
void Context::Finalize() {}
void Context::TearDown() {
    m_pimpl_->m_model_.TearDown();
    m_pimpl_->m_atlas_.TearDown();
}
void Context::SetUp() {
    m_pimpl_->m_model_.SetUp();
    m_pimpl_->m_atlas_.SetUp();
    auto x_box = m_pimpl_->m_model_.GetBoundBox();
    auto i_box = m_pimpl_->m_atlas_.GetIndexBox();
    auto period = m_pimpl_->m_atlas_.GetPeriodicDimension();
    point_type dx;
    dx[0] = (std::get<1>(x_box)[0] - std::get<0>(x_box)[0]) / (std::get<1>(i_box)[0] - std::get<0>(i_box)[0]);
    dx[1] = (std::get<1>(x_box)[1] - std::get<0>(x_box)[1]) / (std::get<1>(i_box)[1] - std::get<0>(i_box)[1]);
    dx[2] = (std::get<1>(x_box)[2] - std::get<0>(x_box)[2]) / (std::get<1>(i_box)[2] - std::get<0>(i_box)[2]);

    for (auto &item : m_pimpl_->m_domains_) { item.second->GetMesh()->GetChart()->SetDx(dx); }
};
void Context::InitializeCondition(Patch *p, Real time_now) {
    for (auto &item : m_pimpl_->m_domains_) {
        item.second->Push(p);
        item.second->InitializeCondition(time_now);
        item.second->Pop(p);
    }
}
void Context::BoundaryCondition(Patch *p, Real time_now, Real time_dt) {
    for (auto &item : m_pimpl_->m_domains_) {
        item.second->Push(p);
        item.second->BoundaryCondition(time_now, time_dt);
        item.second->Pop(p);
    }
}

void Context::Advance(Patch *p, Real time_now, Real time_dt) {
    for (auto &item : m_pimpl_->m_domains_) {
        item.second->Push(p);
        item.second->Advance(time_dt, time_dt);
        item.second->Pop(p);
    }
}

Model &Context::GetModel() const { return m_pimpl_->m_model_; }

Atlas &Context::GetAtlas() const { return m_pimpl_->m_atlas_; }

void Context::RegisterAt(AttributeGroup *attr_grp) {
    for (auto &item : m_pimpl_->m_domains_) { item.second->GetMesh()->RegisterAt(attr_grp); }
}

std::shared_ptr<Domain> Context::SetDomain(std::string const &k, std::shared_ptr<Domain> d) {
    m_pimpl_->m_domains_[k] = d;
    m_pimpl_->m_model_.SetObject(k, d->GetGeoObject());
    return d;
}

std::shared_ptr<Domain> Context::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domains_.find(k);
    return (it == m_pimpl_->m_domains_.end()) ? nullptr : it->second;
}

// std::map<id_type, std::shared_ptr<Patch>> const &Context::GetPatches() const { return m_pimpl_->m_patches_; }
//
// bool Context::RegisterWorker(std::string const &d_name, std::shared_ptr<Domain> const &p) {
//    ASSERT(!IsInitialized());
//
//    auto res = m_pimpl_->m_workers_.emplace(d_name, p);
//    if (!res.second) { res.first->second = p; }
//    db()->Push("Workers/" + d_name, res.first->second->db());
//    return res.second;
//}
// void Context::DeregisterWorker(std::string const &k) {
//    ASSERT(!IsInitialized());
//    m_pimpl_->m_workers_.erase(k);
//}
// std::shared_ptr<Domain> Context::GetWorker(std::string const &d_name) const { return m_pimpl_->m_workers_.at(d_name);
// }
//
// std::map<std::string, std::shared_ptr<Attribute>> const &Context::GetAllAttributes() const {
//    return m_pimpl_->m_global_attributes_;
//};
//
// bool Context::RegisterAttribute(std::string const &key, std::shared_ptr<Attribute> const &v) {
//    ASSERT(!IsInitialized());
//    return m_pimpl_->m_global_attributes_.emplace(key, v).second;
//}
// void Context::DeregisterAttribute(std::string const &key) {
//    ASSERT(!IsInitialized());
//    m_pimpl_->m_global_attributes_.erase(key);
//}
// std::shared_ptr<Attribute> const &Context::GetAttribute(std::string const &key) const {
//    return m_pimpl_->m_global_attributes_.at(key);
//}
//    GetModel().InitializeConditionPatch();
//    GetAtlas().InitializeConditionPatch();
//    auto workers_t = db()->GetTable("Workers");
//    GetModel().GetMaterial().Foreach([]() {
//
//    });
//
//    for (auto const &item : GetModel().GetAllMaterial()) {}
//
//    db()->GetTable("Domains")->Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &v) {
//        if (!v->isTable()) { return; }
//        auto const &t = v->cast_as<data::DataTable>();
//
//        std::shared_ptr m(GLOBAL_MESHVIEW_FACTORY.Create(t.GetTable("MeshBase"),
//                                                         GetModel().AddObject(key,
//                                                         t.GetTable("Geometry")).first));
//
//        m_pimpl_->m_domains_.emplace(key, std::make_shared<Domain>(t.GetTable("Domain"), m));
//
//    });
//    for (auto const &item : GetModel().GetAll()) {
//        auto worker_res = m_pimpl_->m_domains_.emplace(item.first, nullptr);
//        if (worker_res.first->second == nullptr) {
//            worker_res.first->second = std::make_shared<Domain>(workers_t->GetTable(item.first), nullptr,
//            item.second);
//        }
//        workers_t->Link(item.first, worker_res.first->second->db());
//    }
//    std::shared_ptr<geometry::GeoObject> geo = g;
//    if (geo == nullptr) { geo.reset(GLOBAL_GEO_OBJECT_FACTORY.Create(db()->GetTable("Geometry"))); }
//    m_pimpl_->m_chart_.reset(GLOBAL_MESHVIEW_FACTORY.Create(db()->GetTable("MeshBase"), geo));
//
//    m_pimpl_->m_is_initialized_ = true;
//    LOGGER << "Context is initialized!" << std::endl;

//    if (level >= GetAtlas().GetNumOfLevels()) { return; }
//
//    for (auto const &g_item : GetModel().GetAll()) {
//        auto w = m_pimpl_->m_workers_.find(g_item.first);
//        if (w == m_pimpl_->m_workers_.end()) { continue; }
//        for (auto const &mblk : GetAtlas().Level(level)) {
//            if (!g_item.second->CheckOverlap(mblk->GetBoundBox())) { continue; }
//
//            auto p = m_pimpl_->m_patches_[mblk->GetGUID()];
//            if (p == nullptr) {
//                p = std::make_shared<Patch>();
//                //                p->PushMeshBlock(mblk);
//            }
//            w->second->ConvertPatchFromSAMRAI(p);
//            LOGGER << " Domain [ " << std::setw(10) << std::left << w->second->name() << " ] is applied on "
//                   << mblk->GetIndexBox() << " GeoObject id= " << g_item.first << std::endl;
//            w->second->AdvanceData(time_now, dt);
//            m_pimpl_->m_patches_[mblk->GetGUID()] = w->second->ConvertPatchToSAMRAI();
//        }
//    }

}  // namespace engine {
}  // namespace simpla {
