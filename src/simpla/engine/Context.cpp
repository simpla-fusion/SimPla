//
// Created by salmon on 17-2-16.
//
#include "simpla/SIMPLA_config.h"

#include "Context.h"

#include "simpla/data/Data.h"
#include "simpla/engine/Model.h"
#include "simpla/geometry/Chart.h"
#include "simpla/geometry/GeoAlgorithm.h"

#include "Domain.h"
namespace simpla {
namespace engine {

struct Context::pimpl_s {
    std::map<std::string, std::shared_ptr<DomainBase>> m_domains_;
    std::map<std::string, std::shared_ptr<AttributeDesc>> m_global_attributes_;
    std::shared_ptr<geometry::Chart> m_chart_;
    Atlas m_atlas_;
};

Context::Context(std::string const &s_name) : SPObject(s_name), m_pimpl_(new pimpl_s) {}
Context::~Context() { m_pimpl_->m_atlas_.Finalize(); }

std::shared_ptr<data::DataTable> Context::Serialize() const {
    auto res = std::make_shared<data::DataTable>();

    res->SetValue("Name", GetName());
    res->SetValue("Chart", GetChart()->Serialize());
    res->Set("Atlas", m_pimpl_->m_atlas_.Serialize());
    for (auto const &item : m_pimpl_->m_domains_) { res->Link("Domain/" + item.first, item.second->Serialize()); }

    return res;
}
void Context::Deserialize(const std::shared_ptr<data::DataTable> &cfg) {
    DoInitialize();

    SetName(cfg->GetValue<std::string>("Name", "unamed"));

    m_pimpl_->m_atlas_.Deserialize(cfg->GetTable("Atlas"));
    m_pimpl_->m_chart_ = geometry::Chart::Create(cfg->GetTable("Chart"));

    m_pimpl_->m_chart_->SetOrigin(std::get<0>(m_pimpl_->m_atlas_.GetBox()));
    m_pimpl_->m_chart_->SetScale((std::get<1>(m_pimpl_->m_atlas_.GetBox()) - std::get<0>(m_pimpl_->m_atlas_.GetBox())) /
                                 m_pimpl_->m_atlas_.GetDimensions());

    auto t_domain = cfg->GetTable("Domain");
    if (t_domain != nullptr) {
        cfg->GetTable("Domain")->Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &t_cfg) {
            if (t_cfg != nullptr && t_cfg->isTable()) {
                auto p_cfg = std::dynamic_pointer_cast<data::DataTable>(t_cfg);
                std::string s_type = p_cfg->GetValue<std::string>("Type", "Unknown");
                auto res = DomainBase::Create(s_type);
                res->SetName(key);
                res->SetChart(m_pimpl_->m_chart_.get());
                res->Deserialize(p_cfg);
                SetDomain(key, res);
            } else {
                RUNTIME_ERROR << "illegal domain config!" << std::endl;
            }
        });
    }
    Click();
}
//    auto m_cfg = cfg->GetTable("MeshBase");
//    m_cfg->Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &t) {
//    });
//    auto d_cfg = cfg->GetTable("Domains");
//    d_cfg->Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &t) {
//        std::shared_ptr<DomainBase> d = nullptr;
//
//        if (!t->isTable()) {
//            d = DomainBase::Create(t, GetBaseMesh(), nullptr);
//            return;
//        } else {
//            std::shared_ptr<geometry::GeoObject> geo = std::make_shared<geometry::GeoObjectFull>();
//
//            auto t_cfg = std::dynamic_pointer_cast<data::DataTable>(t);
//            geo = GetGeoObject(t_cfg->GetValue<std::string>("Model", ""));
//            auto s_mesh = t_cfg->GetValue<std::string>("MeshBase", "Default");
//            auto p_mesh = m_pimpl_->m_base_mesh_[s_mesh];
//            std::string type = t_cfg->GetValue<std::string>("Type", "");
//            if (type != "") {
//                type = type + "." + p_mesh->GetRegisterName();
//                d = DomainBase::Create(type, p_mesh, geo);
//                d->Deserialize(t_cfg);
//            }
//        }
//        if (d != nullptr) {
//            VERBOSE << "Add DomainBase [" << key << " : " << d->GetRegisterName() << "] " << std::endl;
//            Context::SetDomain(key, d);
//        }
//    });
void Context::DoInitialize() { SPObject::DoInitialize(); }
void Context::DoFinalize() { SPObject::DoFinalize(); }
void Context::DoTearDown() {
    SPObject::DoTearDown();
    m_pimpl_->m_atlas_.TearDown();
}
void Context::DoUpdate() {
    SPObject::DoUpdate();
    m_pimpl_->m_atlas_.Update();
    // TODO: Fix boundary box
    //    m_pimpl_->m_base_mesh_->FitBoundBox(m_pimpl_->m_model_.GetBoundBox());
    for (auto &d : m_pimpl_->m_domains_) { d.second->Update(); }
}

Atlas &Context::GetAtlas() const { return m_pimpl_->m_atlas_; }

void Context::SetChart(std::shared_ptr<geometry::Chart> const &c) { m_pimpl_->m_chart_ = c; }
geometry::Chart const *Context::GetChart() const { return m_pimpl_->m_chart_.get(); }

void Context::SetDomain(std::string const &s_name, std::shared_ptr<DomainBase> const &d) {
    m_pimpl_->m_domains_[s_name] = d;
}

std::shared_ptr<DomainBase> Context::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domains_.find(k);
    return (it == m_pimpl_->m_domains_.end()) ? nullptr : it->second;
}

std::map<std::string, std::shared_ptr<AttributeDesc>> Context::CollectRegisteredAttributes() const {
    std::map<std::string, std::shared_ptr<AttributeDesc>> m_global_attributes_;
    for (auto const &item : GetAllDomains()) { item.second->RegisterDescription(&m_global_attributes_); }
    return m_global_attributes_;
}

std::map<std::string, std::shared_ptr<DomainBase>> &Context::GetAllDomains() { return m_pimpl_->m_domains_; }

std::map<std::string, std::shared_ptr<DomainBase>> const &Context::GetAllDomains() const {
    return m_pimpl_->m_domains_;
}

void Context::InitialCondition(Patch *patch, Real time_now) {
    Update();
    for (auto &d : GetAllDomains()) { d.second->InitialCondition(patch, time_now); }
}
void Context::BoundaryCondition(Patch *patch, Real time_now, Real time_dt) {
    Update();
    for (auto &d : GetAllDomains()) { d.second->BoundaryCondition(patch, time_now, time_dt); }
}
void Context::Advance(Patch *patch, Real time_now, Real time_dt) {
    Update();
    for (auto &d : GetAllDomains()) { d.second->Advance(patch, time_now, time_dt); }
}

// std::map<id_type, std::shared_ptr<Patch>> const &Context::GetPatches() const { return m_pimpl_->m_patches_; }
//
// bool Context::RegisterWorker(std::string const &d_name, std::shared_ptr<DomainBase> const &p) {
//    ASSERT(!IsInitialized());
//
//    auto res = m_pimpl_->m_workers_.emplace(d_name, p);
//    if (!res.second) { res.first->second = p; }
//    db()->Deserialize("Workers/" + d_name, res.first->second->db());
//    return res.second;
//}
// void Context::DeregisterWorker(std::string const &k) {
//    ASSERT(!IsInitialized());
//    m_pimpl_->m_workers_.erase(k);
//}
// std::shared_ptr<DomainBase> Context::GetWorker(std::string const &d_name) const { return
// m_pimpl_->m_workers_.at(d_name);
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
//        m_pimpl_->m_domains_.emplace(key, std::make_shared<DomainBase>(t.GetTable("DomainBase"), m));
//
//    });
//    for (auto const &item : GetModel().GetAllAttributes()) {
//        auto worker_res = m_pimpl_->m_domains_.emplace(item.first, nullptr);
//        if (worker_res.first->second == nullptr) {
//            worker_res.first->second = std::make_shared<DomainBase>(workers_t->GetTable(item.first), nullptr,
//            item.second);
//        }
//        workers_t->Link(item.first, worker_res.first->second->db());
//    }
//    std::shared_ptr<geometry::GeoObject> geo = g;
//    if (geo == nullptr) { geo.reset(GLOBAL_GEO_OBJECT_FACTORY.Create(db()->GetTable("Geometry"))); }
//    m_pimpl_->m_base_chart_.reset(GLOBAL_MESHVIEW_FACTORY.Create(db()->GetTable("MeshBase"), geo));
//
//    m_pimpl_->m_is_initialized_ = true;
//    LOGGER << "Context is initialized!" << std::endl;

//    if (level >= GetAtlas().GetNumOfLevels()) { return; }
//
//    for (auto const &g_item : GetModel().GetAllAttributes()) {
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
//            LOGGER << " DomainBase [ " << std::setw(10) << std::left << w->second->name() << " ] is applied on "
//                   << mblk->GetIndexBox() << " GeoObject id= " << g_item.first << std::endl;
//            w->second->AdvanceData(time_now, dt);
//            m_pimpl_->m_patches_[mblk->GetGUID()] = w->second->ConvertPatchToSAMRAI();
//        }
//    }

}  // namespace engine {
}  // namespace simpla {
