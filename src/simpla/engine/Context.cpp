//
// Created by salmon on 17-2-16.
//
#include "Context.h"
#include <simpla/data/all.h>
#include <simpla/geometry/GeoAlgorithm.h>
#include "Domain.h"
#include "MeshBase.h"
namespace simpla {
namespace engine {

struct Context::pimpl_s {
    pimpl_s() {}

    std::map<std::string, std::shared_ptr<AttributeDesc>> m_global_attributes_;
    std::map<std::string, std::shared_ptr<Domain>> m_domains_;
    Model m_model_;
    Atlas m_atlas_;
};

Context::Context(std::string const &s_name) : m_pimpl_(new pimpl_s), SPObject(s_name) {}
Context::~Context() {}

std::shared_ptr<data::DataTable> Context::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    for (auto const &item : m_pimpl_->m_domains_) { res->Link("Domains/" + item.first, item.second->Serialize()); }
    res->Set("Model", m_pimpl_->m_model_.Serialize());
    res->Set("Atlas", m_pimpl_->m_atlas_.Serialize());
    return res;
}
void Context::Deserialize(const std::shared_ptr<DataTable> &cfg) {
    m_pimpl_->m_atlas_.Deserialize(cfg->GetTable("Atlas"));
    m_pimpl_->m_model_.Deserialize(cfg->GetTable("Model"));

    auto d_cfg = cfg->GetTable("Domains");
    d_cfg->Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &t) {
        auto geo = m_pimpl_->m_model_.GetObject(key);

        auto d = Domain::Create(t, key, (geo != nullptr) ? geo : std::make_shared<geometry::GeoObjectFull>());
        if (d != nullptr) {
            VERBOSE << "Add Domain [" << key << " : " << d->GetRegisterName() << "] " << std::endl;
            d->Initialize();
            Context::SetDomain(key, d);
        }
    });
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

    for (auto &item : m_pimpl_->m_domains_) {
        item.second->Initialize();
        item.second->GetMesh()->SetOrigin(std::get<0>(x_box));
        item.second->GetMesh()->SetDx(dx);
        item.second->RegisterDescription(&m_pimpl_->m_global_attributes_);
        item.second->SetUp();
    }
};
// std::shared_ptr<Patch> Context::ApplyInitializeCondition(const std::shared_ptr<Patch> &p, Real time_now) {
//    std::shared_ptr<Patch> res = p;
//    for (auto &item : m_pimpl_->m_domains_) { res = item.second->DoInitialCondition(res, time_now); }
//    return res;
//}
// std::shared_ptr<Patch> Context::DoBoundaryCondition(const std::shared_ptr<Patch> &p, Real time_now, Real time_dt) {
//    std::shared_ptr<Patch> res = p;
//    for (auto &item : m_pimpl_->m_domains_) { res = item.second->DoBoundaryCondition(res, time_now, time_dt); }
//    return res;
//}
//
// std::shared_ptr<Patch> Context::DoAdvance(const std::shared_ptr<Patch> &p, Real time_now, Real time_dt) {
//    std::shared_ptr<Patch> res = p;
//    for (auto &item : m_pimpl_->m_domains_) { res = item.second->DoAdvance(res, time_now, time_dt); }
//    return res;
//}

Model &Context::GetModel() const { return m_pimpl_->m_model_; }

Atlas &Context::GetAtlas() const { return m_pimpl_->m_atlas_; }

std::map<std::string, std::shared_ptr<AttributeDesc>> const &Context::GetRegisteredAttribute() const {
    return m_pimpl_->m_global_attributes_;
}

std::shared_ptr<Domain> Context::SetDomain(std::string const &k, std::shared_ptr<Domain> d) {
    if (d == nullptr) { return d; }
    auto res = m_pimpl_->m_domains_.emplace(k, d);
    if (res.first->second == nullptr) {
        res.first->second = d;
    } else if (!res.second) {
        RUNTIME_ERROR << "Re-assign domain " << k << std::endl;
    }
    return res.first->second;
}

std::shared_ptr<Domain> Context::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domains_.find(k);
    return (it == m_pimpl_->m_domains_.end()) ? nullptr : it->second;
}
std::map<std::string, std::shared_ptr<Domain>> &Context::GetAllDomains() { return m_pimpl_->m_domains_; };
std::map<std::string, std::shared_ptr<Domain>> const &Context::GetAllDomain() const { return m_pimpl_->m_domains_; };
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
