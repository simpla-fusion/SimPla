//
// Created by salmon on 17-2-16.
//
#include "Context.h"
#include "Domain.h"
#include "Mesh.h"
#include "Task.h"
#include "Worker.h"
#include "simpla/data/all.h"
#include "simpla/geometry/GeoAlgorithm.h"

namespace simpla {
namespace engine {

struct Context::pimpl_s {
    std::map<std::string, std::shared_ptr<Worker>> m_workers_;
    std::map<std::string, std::shared_ptr<Attribute>> m_global_attributes_;
    std::map<std::string, std::shared_ptr<Domain>> m_domain_;
    Model m_model_;
    Atlas m_atlas_;
};

Context::Context() : m_pimpl_(new pimpl_s) {}
Context::~Context() {}
Atlas &Context::GetAtlas() const { return m_pimpl_->m_atlas_; }

std::shared_ptr<data::DataTable> Context::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    for (auto const &item : m_pimpl_->m_domain_) { res->Link("Domain/" + item.first, item.second->Serialize()); }
    return res;
}
void Context::Deserialize(std::shared_ptr<DataTable> cfg) {
    auto domain_t = cfg->GetTable("Domains");
    if (domain_t == nullptr) { return; }
    domain_t->Foreach([&](std::string const &k, std::shared_ptr<DataEntity> t) {
        auto v = std::make_shared<Domain>();
        if (t->isTable()) { v->Deserialize(std::dynamic_pointer_cast<data::DataTable>(t)); }
        SetDomain(k, v);
    });
}

void Context::Initialize() {}
void Context::Finalize() {}
void Context::TearDown() {
    m_pimpl_->m_model_.TearDown();
    m_pimpl_->m_atlas_.TearDown();
}
void Context::SetUp() {
    for (auto &item : m_pimpl_->m_domain_) { m_pimpl_->m_model_.AddObject(item.first, item.second->GetGeoObject()); }
    m_pimpl_->m_model_.SetUp();
};
void Context::SetUpDataOnPatch(Patch *p, Real time_now) {
    for (auto &item : m_pimpl_->m_domain_) { item.second->SetUpDataOnPatch(p, time_now); }
}
void Context::UpdateDataOnPatch(Patch *p, Real time_now, Real time_dt) {
    for (auto &item : m_pimpl_->m_domain_) { item.second->UpdateDataOnPatch(p, time_now, time_dt); }
}

Model &Context::GetModel() const { return m_pimpl_->m_model_; }

void Context::Register(AttributeGroup *attr_grp) {
    for (auto &item : m_pimpl_->m_domain_) { item.second->Register(attr_grp); }
}
void Context::SetDomain(std::string const &k, std::shared_ptr<Domain> d) {
    auto res = m_pimpl_->m_domain_.emplace(k, d);
    if (!res.second) { res.first->second = d; }
}
std::shared_ptr<Domain> Context::GetDomain(std::string const &k) {
    auto res = m_pimpl_->m_domain_.emplace(k, nullptr);
    if (res.first->second == nullptr) { res.first->second = std::make_shared<Domain>(); }
    return res.first->second;
}
std::shared_ptr<Domain> Context::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domain_.find(k);
    return (it == m_pimpl_->m_domain_.end()) ? nullptr : it->second;
}
// std::map<id_type, std::shared_ptr<Patch>> const &Context::GetPatches() const { return m_pimpl_->m_patches_; }
//
// bool Context::RegisterWorker(std::string const &d_name, std::shared_ptr<Worker> const &p) {
//    ASSERT(!IsInitialized());
//
//    auto res = m_pimpl_->m_workers_.emplace(d_name, p);
//    if (!res.second) { res.first->second = p; }
//    db()->Set("Workers/" + d_name, res.first->second->db());
//    return res.second;
//}
// void Context::DeregisterWorker(std::string const &k) {
//    ASSERT(!IsInitialized());
//    m_pimpl_->m_workers_.erase(k);
//}
// std::shared_ptr<Worker> Context::GetWorker(std::string const &d_name) const { return m_pimpl_->m_workers_.at(d_name);
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
//    GetModel().SetUpDataOnPatch();
//    GetAtlas().SetUpDataOnPatch();
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
//        std::shared_ptr m(GLOBAL_MESHVIEW_FACTORY.Create(t.GetTable("Mesh"),
//                                                         GetModel().AddObject(key,
//                                                         t.GetTable("Geometry")).first));
//
//        m_pimpl_->m_worker_.emplace(key, std::make_shared<Worker>(t.GetTable("Worker"), m));
//
//    });
//    for (auto const &item : GetModel().GetAll()) {
//        auto worker_res = m_pimpl_->m_worker_.emplace(item.first, nullptr);
//        if (worker_res.first->second == nullptr) {
//            worker_res.first->second = std::make_shared<Worker>(workers_t->GetTable(item.first), nullptr,
//            item.second);
//        }
//        workers_t->Link(item.first, worker_res.first->second->db());
//    }
//    std::shared_ptr<geometry::GeoObject> geo = g;
//    if (geo == nullptr) { geo.reset(GLOBAL_GEO_OBJECT_FACTORY.Create(db()->GetTable("Geometry"))); }
//    m_pimpl_->m_chart_.reset(GLOBAL_MESHVIEW_FACTORY.Create(db()->GetTable("Mesh"), geo));
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
//            w->second->Push(p);
//            LOGGER << " Worker [ " << std::setw(10) << std::left << w->second->name() << " ] is applied on "
//                   << mblk->GetIndexBox() << " GeoObject id= " << g_item.first << std::endl;
//            w->second->Advance(time_now, dt);
//            m_pimpl_->m_patches_[mblk->GetGUID()] = w->second->Pop();
//        }
//    }

}  // namespace engine {
}  // namespace simpla {
