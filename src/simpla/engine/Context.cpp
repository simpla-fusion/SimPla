//
// Created by salmon on 17-2-16.
//
#include "simpla/SIMPLA_config.h"

#include "simpla/data/Data.h"
#include "simpla/data/EnableCreateFromDataTable.h"
#include "simpla/engine/Model.h"
#include "simpla/geometry/Chart.h"
#include "simpla/geometry/GeoAlgorithm.h"

#include "Context.h"
#include "Domain.h"
#include "Mesh.h"
#include "simpla/geometry/BoxUtilities.h"
namespace simpla {
namespace engine {

struct Context::pimpl_s {
    std::map<std::string, std::shared_ptr<Model>> m_models_;
    std::shared_ptr<MeshBase> m_mesh_;

    std::map<std::string, std::shared_ptr<DomainBase>> m_domains_;
    std::map<std::string, std::shared_ptr<AttributeDesc>> m_global_attributes_;

    index_box_type m_bound_index_box_{{0, 0, 0}, {1, 1, 1}};
    box_type m_bound_box_{{0, 0, 0}, {1, 1, 1}};
};

Context::Context(std::string const &s_name) : SPObject(s_name), m_pimpl_(new pimpl_s) {}

Context::~Context() {}

std::shared_ptr<data::DataTable> Context::Serialize() const {
    auto res = std::make_shared<data::DataTable>();

    res->SetValue("Name", GetName());
    res->SetValue("Mesh", GetMesh()->Serialize());
    for (auto const &item : m_pimpl_->m_domains_) { res->Link("Domain/" + item.first, item.second->Serialize()); }
    for (auto const &item : m_pimpl_->m_models_) { res->Link("Model/" + item.first, item.second->Serialize()); }

    return res;
}

void Context::Deserialize(const std::shared_ptr<data::DataTable> &cfg) {
    DoInitialize();

    m_pimpl_->m_mesh_ = MeshBase::Create(cfg->GetTable("Mesh"));

    auto t_model = cfg->GetTable("Model");

    if (t_model != nullptr) {
        t_model->Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &t_cfg) {
            m_pimpl_->m_models_[key] = Model::Create(std::dynamic_pointer_cast<data::DataTable>(t_cfg));
        });
    }

    auto t_domain = cfg->GetTable("Domains");
    if (t_domain != nullptr) {
        t_domain->Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &t_cfg) {
            if (t_cfg != nullptr && t_cfg->isTable()) {
                auto p_cfg = std::dynamic_pointer_cast<data::DataTable>(t_cfg);

                m_pimpl_->m_domains_[key] =
                    DomainBase::Create(p_cfg, GetMesh(), GetModel(p_cfg->GetValue<std::string>("Model", "")));

                m_pimpl_->m_domains_[key]->SetName(key);

            } else {
                RUNTIME_ERROR << "illegal domain config!" << std::endl;
            }
        });
    }
    std::get<0>(m_pimpl_->m_bound_box_) = cfg->GetValue("Mesh/Box/lo", point_type{0, 0, 0});
    std::get<1>(m_pimpl_->m_bound_box_) = cfg->GetValue("Mesh/Box/hi", point_type{0, 0, 0});
    std::get<1>(m_pimpl_->m_bound_index_box_) = cfg->GetValue("Mesh/Dimensions", nTuple<int, 3>{1, 1, 1});

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
//            auto p_mesh = m_pack_->m_base_mesh_[s_mesh];
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

box_type Context::GetBoundBox() const { return m_pimpl_->m_bound_box_; }
index_box_type Context::GetIndexBox() const { return m_pimpl_->m_bound_index_box_; }
void Context::DoInitialize() { SPObject::DoInitialize(); }
void Context::DoFinalize() { SPObject::DoFinalize(); }
void Context::DoTearDown() { SPObject::DoTearDown(); }
void Context::DoUpdate() {
    SPObject::DoUpdate();

    GetMesh()->GetChart()->SetOrigin(std::get<0>(GetBoundBox()));
    for (auto &d : m_pimpl_->m_domains_) { d.second->Update(); }

    //    auto scale = GetMesh()->GetChart()->GetScale();
    //    auto ib = GetAllDomains().begin();
    //    auto ie = GetAllDomains().end();
    //
    //    box_type bound_box = ib->second->GetBoundary()->GetBoundBox();
    //    ++ib;
    //    for (; ib != ie; ++ib) { bound_box = geometry::expand(bound_box, ib->second->GetBoundary()->GetBoundBox()); }
    //    auto bound_box = GetBoundBox();
    //    return index_box_type{std::get<0>(GetMesh()->GetChart()->invert_local_coordinates(std::get<0>(bound_box))),
    //                          std::get<0>(GetMesh()->GetChart()->invert_local_coordinates(std::get<1>(bound_box)))};
}

void Context::SetMesh(std::shared_ptr<MeshBase> const &m) { m_pimpl_->m_mesh_ = m; }
MeshBase const *Context::GetMesh() const { return m_pimpl_->m_mesh_.get(); }
MeshBase *Context::GetMesh() { return m_pimpl_->m_mesh_.get(); }

void Context::SetModel(std::string const &k, std::shared_ptr<Model> const &m) const { m_pimpl_->m_models_[k] = m; }
std::shared_ptr<Model> Context::GetModel(std::string const &k) const {
    auto it = m_pimpl_->m_models_.find(k);
    return it == m_pimpl_->m_models_.end() ? nullptr : it->second;
}

std::shared_ptr<DomainBase> Context::CreateDomain(std::string const &k, std::shared_ptr<data::DataTable> const &t) {
    auto res = DomainBase::Create(t, GetMesh(), GetModel(t->GetValue<std::string>("Model", k)));
    SetDomain(k, res);
    return res;
};
void Context::SetDomain(std::string const &k, std::shared_ptr<DomainBase> const &d) { m_pimpl_->m_domains_[k] = d; }

std::shared_ptr<DomainBase> Context::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domains_.find(k);
    return (it == m_pimpl_->m_domains_.end()) ? nullptr : it->second;
}

// std::map<std::string, std::shared_ptr<AttributeDesc>> Context::CollectRegisteredAttributes() const {
//    std::map<std::string, std::shared_ptr<AttributeDesc>> m_global_attributes_;
//    GetMesh()->RegisterDescription(&m_global_attributes_);
//    return m_global_attributes_;
//}

std::map<std::string, std::shared_ptr<DomainBase>> &Context::GetAllDomains() { return m_pimpl_->m_domains_; }

std::map<std::string, std::shared_ptr<DomainBase>> const &Context::GetAllDomains() const {
    return m_pimpl_->m_domains_;
}

void Context::Pull(Patch *p) { GetMesh()->Pull(p); };
void Context::Push(Patch *p) { GetMesh()->Push(p); };

void Context::InitialCondition(Real time_now) {
//    VERBOSE << "InitialCondition   \t:" << GetName() << std::endl;
    GetMesh()->InitialCondition(time_now);
    for (auto &d : GetAllDomains()) { d.second->InitialCondition(time_now); }
}
void Context::BoundaryCondition(Real time_now, Real dt) {
//    VERBOSE << "Boundary Condition \t:" << GetName() << std::endl;
    GetMesh()->BoundaryCondition(time_now, dt);
    for (auto &d : GetAllDomains()) { d.second->BoundaryCondition(time_now, dt); }
}
void Context::Advance(Real time_now, Real dt) {
//    VERBOSE << "Advance            \t:" << GetName() << std::endl;
    GetMesh()->Advance(time_now, dt);
    for (auto &d : GetAllDomains()) { d.second->Advance(time_now, dt); }
}

void Context::TagRefinementCells(Real time_now) {
//    VERBOSE << "TagRefinementCells  \t:" << GetName() << std::endl;
    GetMesh()->TagRefinementCells(time_now);
    for (auto &d : GetAllDomains()) { d.second->TagRefinementCells(time_now); }
}

// std::map<id_type, std::shared_ptr<Patch>> const &Context::GetPatches() const { return m_pack_->m_patches_; }
//
// bool Context::RegisterWorker(std::string const &d_name, std::shared_ptr<DomainBase> const &p) {
//    ASSERT(!IsInitialized());
//
//    auto res = m_pack_->m_workers_.emplace(d_name, p);
//    if (!res.second) { res.first->second = p; }
//    db()->Deserialize("Workers/" + d_name, res.first->second->db());
//    return res.second;
//}
// void Context::DeregisterWorker(std::string const &k) {
//    ASSERT(!IsInitialized());
//    m_pack_->m_workers_.erase(k);
//}
// std::shared_ptr<DomainBase> Context::GetWorker(std::string const &d_name) const { return
// m_pack_->m_workers_.at(d_name);
// }
//
// std::map<std::string, std::shared_ptr<Attribute>> const &Context::GetAllAttributes() const {
//    return m_pack_->m_global_attributes_;
//};
//
// bool Context::RegisterAttribute(std::string const &key, std::shared_ptr<Attribute> const &v) {
//    ASSERT(!IsInitialized());
//    return m_pack_->m_global_attributes_.emplace(key, v).second;
//}
// void Context::DeregisterAttribute(std::string const &key) {
//    ASSERT(!IsInitialized());
//    m_pack_->m_global_attributes_.erase(key);
//}
// std::shared_ptr<Attribute> const &Context::GetAttribute(std::string const &key) const {
//    return m_pack_->m_global_attributes_.at(key);
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
//        m_pack_->m_domains_.emplace(key, std::make_shared<DomainBase>(t.GetTable("DomainBase"), m));
//
//    });
//    for (auto const &item : GetModel().GetAllAttributes()) {
//        auto worker_res = m_pack_->m_domains_.emplace(item.first, nullptr);
//        if (worker_res.first->second == nullptr) {
//            worker_res.first->second = std::make_shared<DomainBase>(workers_t->GetTable(item.first), nullptr,
//            item.second);
//        }
//        workers_t->Link(item.first, worker_res.first->second->db());
//    }
//    std::shared_ptr<geometry::GeoObject> geo = g;
//    if (geo == nullptr) { geo.reset(GLOBAL_GEO_OBJECT_FACTORY.Create(db()->GetTable("Geometry"))); }
//    m_pack_->m_base_chart_.reset(GLOBAL_MESHVIEW_FACTORY.Create(db()->GetTable("MeshBase"), geo));
//
//    m_pack_->m_is_initialized_ = true;
//    LOGGER << "Context is initialized!" << std::endl;

//    if (level >= GetAtlas().GetNumOfLevels()) { return; }
//
//    for (auto const &g_item : GetModel().GetAllAttributes()) {
//        auto w = m_pack_->m_workers_.find(g_item.first);
//        if (w == m_pack_->m_workers_.end()) { continue; }
//        for (auto const &mblk : GetAtlas().Level(level)) {
//            if (!g_item.second->CheckOverlap(mblk->GetBoundBox())) { continue; }
//
//            auto p = m_pack_->m_patches_[mblk->GetID()];
//            if (p == nullptr) {
//                p = std::make_shared<Patch>();
//                //                p->PushMeshBlock(mblk);
//            }
//            w->second->GetPatch(p);
//            LOGGER << " DomainBase [ " << std::setw(10) << std::left << w->second->name() << " ] is applied on "
//                   << mblk->GetIndexBox() << " GeoObject id= " << g_item.first << std::endl;
//            w->second->AdvanceData(time_now, dt);
//            m_pack_->m_patches_[mblk->GetID()] = w->second->PushPatch();
//        }
//    }

}  // namespace engine {
}  // namespace simpla {
