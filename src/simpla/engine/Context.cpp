//
// Created by salmon on 17-2-16.
//
#include "Context.h"
#include <simpla/data/all.h>
#include <simpla/geometry/Chart.h>
#include <simpla/geometry/GeoAlgorithm.h>
#include "Domain.h"
#include "MeshBase.h"
namespace simpla {
namespace engine {

struct Context::pimpl_s {
    std::map<std::string, std::shared_ptr<geometry::Chart>> m_chart_;
    std::map<std::string, std::shared_ptr<geometry::GeoObject>> m_geo_obj_;

    std::map<std::string, std::shared_ptr<MeshBase>> m_mesh_;
    std::map<std::string, std::shared_ptr<Domain>> m_domains_;
    std::map<std::string, std::shared_ptr<AttributeDesc>> m_global_attributes_;

    Atlas m_atlas_;
    box_type m_bound_box_;
};

Context::Context(std::string const &s_name) : SPObject(s_name), m_pimpl_(new pimpl_s) {}
Context::~Context() {}

std::shared_ptr<data::DataTable> Context::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    for (auto const &item : m_pimpl_->m_domains_) { res->Link("Domains/" + item.first, item.second->Serialize()); }

    res->Set("Atlas", m_pimpl_->m_atlas_.Serialize());

    return res;
}
void Context::Deserialize(const std::shared_ptr<DataTable> &cfg) {
    DoInitialize();
    m_pimpl_->m_atlas_.Deserialize(cfg->GetTable("Atlas"));

    auto t_mesh = cfg->GetTable("Mesh");
    if (t_mesh != nullptr) {
        t_mesh->Foreach(
            [&](std::string const &key, std::shared_ptr<data::DataEntity> const &t_cfg) { SetMesh(key, t_cfg); });
    }

    auto t_model = cfg->GetTable("Model");
    if (t_model != nullptr) {
        t_model->Foreach(
            [&](std::string const &key, std::shared_ptr<data::DataEntity> const &t_cfg) { SetGeoObject(key, t_cfg); });
    }
    auto t_domain = cfg->GetTable("Domain");
    if (t_domain != nullptr) {
        cfg->GetTable("Domain")->Foreach(
            [&](std::string const &key, std::shared_ptr<data::DataEntity> const &t_cfg) { SetDomain(key, t_cfg); });
    }
    //    auto m_cfg = cfg->GetTable("Mesh");
    //    m_cfg->Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &t) {
    //        // FIXME: !!!!!
    //    });
    //    auto d_cfg = cfg->GetTable("Domains");
    //    d_cfg->Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &t) {
    //        std::shared_ptr<Domain> d = nullptr;
    //
    //        if (!t->isTable()) {
    //            d = Domain::Create(t, GetMesh(), nullptr);
    //            return;
    //        } else {
    //            std::shared_ptr<geometry::GeoObject> geo = std::make_shared<geometry::GeoObjectFull>();
    //
    //            auto t_cfg = std::dynamic_pointer_cast<data::DataTable>(t);
    //            geo = GetGeoObject(t_cfg->GetValue<std::string>("Model", ""));
    //            auto s_mesh = t_cfg->GetValue<std::string>("Mesh", "Default");
    //            auto p_mesh = m_pimpl_->m_mesh_[s_mesh];
    //            std::string type = t_cfg->GetValue<std::string>("Type", "");
    //            if (type != "") {
    //                type = type + "." + p_mesh->GetRegisterName();
    //                d = Domain::Create(type, p_mesh, geo);
    //                d->Deserialize(t_cfg);
    //            }
    //        }
    //        if (d != nullptr) {
    //            VERBOSE << "Add Domain [" << key << " : " << d->GetRegisterName() << "] " << std::endl;
    //            Context::SetDomain(key, d);
    //        }
    //    });
}

void Context::Initialize() {}
void Context::Finalize() {}
void Context::TearDown() { m_pimpl_->m_atlas_.TearDown(); }
void Context::Update() {
    m_pimpl_->m_atlas_.Update();
    for (auto &item : m_pimpl_->m_mesh_) { item.second->RegisterDescription(&m_pimpl_->m_global_attributes_); }
};

Atlas &Context::GetAtlas() const { return m_pimpl_->m_atlas_; }
int Context::GetNDims() const { return 3; };
box_type Context::GetBoundBox() const { return m_pimpl_->m_bound_box_; };

void Context::SetGeoObject(std::string const &s_name, std::shared_ptr<geometry::GeoObject> const &g) {
    m_pimpl_->m_geo_obj_[s_name] = g;
}
void Context::SetGeoObject(std::string const &s_name, std::shared_ptr<data::DataEntity> const &cfg) {
    SetGeoObject(s_name, GetGeoObject(cfg));
}
std::shared_ptr<geometry::GeoObject> Context::GetGeoObject(std::shared_ptr<data::DataEntity> const &cfg) const {
    std::shared_ptr<geometry::GeoObject> res = nullptr;
    if (cfg == nullptr) {
    } else if (cfg->isTable()) {
        res = geometry::GeoObject::Create(cfg);
    } else if (cfg->value_type_info() == typeid(std::string)) {
        res = GetGeoObject(data::data_cast<std::string>(cfg));
    }
    return res;
};
std::shared_ptr<geometry::GeoObject> Context::GetGeoObject(std::string const &s_name) const {
    auto it = m_pimpl_->m_geo_obj_.find(s_name);
    return it != m_pimpl_->m_geo_obj_.end() ? it->second : nullptr;
}

void Context::SetMesh(std::string const &s_name, std::shared_ptr<MeshBase> const &m) { m_pimpl_->m_mesh_[s_name] = m; }
void Context::SetMesh(std::string const &s_name, std::shared_ptr<data::DataEntity> const &cfg) {
    SetMesh(s_name, GetMesh(cfg));
}
std::shared_ptr<MeshBase> Context::GetMesh(std::string const &k) const {
    auto it = m_pimpl_->m_mesh_.find(k);
    return it != m_pimpl_->m_mesh_.end() ? it->second : nullptr;
}

std::shared_ptr<MeshBase> Context::GetMesh(std::shared_ptr<data::DataEntity> const &cfg) const {
    std::shared_ptr<MeshBase> res = nullptr;
    if (cfg == nullptr) {
        res = GetMesh("Default");
    } else if (cfg->value_type_info() == typeid(std::string)) {
        res = GetMesh(data::data_cast<std::string>(cfg));
    } else if (cfg->isTable()) {
        res = MeshBase::Create(cfg);
    } else {
        RUNTIME_ERROR << "illegal domain config!" << std::endl;
    }
    return res;
}

void Context::SetDomain(std::string const &s_name, std::shared_ptr<Domain> const &d) {
    m_pimpl_->m_domains_[s_name] = d;
}
void Context::SetDomain(std::string const &k, std::shared_ptr<data::DataEntity> const &cfg) {
    SetDomain(k, GetDomain(cfg));
};
std::shared_ptr<Domain> Context::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domains_.find(k);
    return (it == m_pimpl_->m_domains_.end()) ? nullptr : it->second;
}

std::shared_ptr<Domain> Context::GetDomain(std::shared_ptr<data::DataEntity> const &cfg) const {
    std::shared_ptr<Domain> res = nullptr;
    if (cfg == nullptr) {
    } else if (cfg->isTable()) {
        auto t_cfg = std::dynamic_pointer_cast<data::DataTable>(cfg);
        res = Domain::Create(cfg, GetMesh(t_cfg->Get("Mesh")), GetGeoObject(t_cfg->Get("Model")));
    } else {
        RUNTIME_ERROR << "illegal domain config!" << std::endl;
    }
    return res;
}

std::map<std::string, std::shared_ptr<AttributeDesc>> const &Context::GetRegisteredAttribute() const {
    return m_pimpl_->m_global_attributes_;
}

std::map<std::string, std::shared_ptr<Domain>> &Context::GetAllDomains() { return m_pimpl_->m_domains_; };
std::map<std::string, std::shared_ptr<Domain>> const &Context::GetAllDomains() const { return m_pimpl_->m_domains_; };

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
//    for (auto const &item : GetModel().GetAllAttributes()) {
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
//            LOGGER << " Domain [ " << std::setw(10) << std::left << w->second->name() << " ] is applied on "
//                   << mblk->GetIndexBox() << " GeoObject id= " << g_item.first << std::endl;
//            w->second->AdvanceData(time_now, dt);
//            m_pimpl_->m_patches_[mblk->GetGUID()] = w->second->ConvertPatchToSAMRAI();
//        }
//    }

}  // namespace engine {
}  // namespace simpla {
