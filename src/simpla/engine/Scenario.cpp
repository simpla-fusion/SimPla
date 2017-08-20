//
// Created by salmon on 17-8-20.
//

#include "simpla/SIMPLA_config.h"

#include "Atlas.h"
#include "Domain.h"
#include "Mesh.h"
#include "Model.h"
#include "Scenario.h"
#include "Schedule.h"
#include "simpla/data/DataNode.h"
namespace simpla {
namespace engine {

struct Scenario::pimpl_s {
    std::shared_ptr<MeshBase> m_mesh_;
    std::shared_ptr<Atlas> m_atlas_;
    std::shared_ptr<Schedule> m_schedule_;
    std::map<std::string, std::shared_ptr<Model>> m_models_;
    std::map<std::string, std::shared_ptr<DomainBase>> m_domains_;
};

Scenario::Scenario() : m_pimpl_(new pimpl_s) {}
Scenario::~Scenario() { delete m_pimpl_; }

void Scenario::Serialize(std::shared_ptr<data::DataNode> const &cfg) const {
    base_type::Serialize(cfg);

    GetMesh()->Serialize(cfg->NewNode("Mesh"));
    GetAtlas()->Serialize(cfg->NewNode("Atlas"));
    GetSchedule()->Serialize(cfg->NewNode("Schedule"));

    auto model = cfg->NewNode("Model");
    for (auto const &item : m_pimpl_->m_models_) { item.second->Serialize(model->NewNode(item.first)); }

    auto domain = cfg->NewNode("Domain");
    for (auto const &item : m_pimpl_->m_domains_) { item.second->Serialize(domain->NewNode(item.first)); }
}

void Scenario::Deserialize(std::shared_ptr<const data::DataNode> const &cfg) {
    DoInitialize();
    base_type::Deserialize(cfg);
    SetMesh(MeshBase::New(cfg->FindNode("Mesh")));
    SetAtlas(Atlas::New(cfg->FindNode("Atlas")));

    if (auto model = cfg->FindNode("Model")) {
        for (auto p = model->FirstChild(); p != nullptr; p = p->Next()) { SetModel(p->Key(), Model::New(p)); }
    }
    if (auto domain = cfg->FindNode("Domain")) {
        for (auto p = domain->FirstChild(); p != nullptr; p = p->Next()) {
            auto key = p->GetValue<std::string>("Name", "");
            auto m = key.empty() ? nullptr : m_pimpl_->m_models_.at(key);
            SetDomain(key, DomainBase::New(m_pimpl_->m_mesh_, m));
        }
    }
    SetSchedule(Schedule::New(cfg->FindNode("Schedule"), shared_from_this()));

    Click();
}

void Scenario::DoInitialize() { SPObject::DoInitialize(); }
void Scenario::DoFinalize() { SPObject::DoFinalize(); }
void Scenario::DoTearDown() { SPObject::DoTearDown(); }
void Scenario::DoUpdate() { SPObject::DoUpdate(); }

void Scenario::SetAtlas(std::shared_ptr<Atlas> const &m) { m_pimpl_->m_atlas_ = m; }
std::shared_ptr<Atlas> Scenario::GetAtlas() const { return m_pimpl_->m_atlas_; }

void Scenario::SetMesh(std::shared_ptr<MeshBase> const &m) { m_pimpl_->m_mesh_ = m; }
std::shared_ptr<MeshBase> Scenario::GetMesh() const { return m_pimpl_->m_mesh_; }

void Scenario::SetModel(std::string const &k, std::shared_ptr<Model> const &m) { m_pimpl_->m_models_[k] = m; }
std::shared_ptr<const Model> Scenario::GetModel(std::string const &k) const {
    auto it = m_pimpl_->m_models_.find(k);
    return it == m_pimpl_->m_models_.end() ? nullptr : it->second;
}

std::shared_ptr<DomainBase> Scenario::NewDomain(std::string const &k, const std::shared_ptr<DataNode> &t) {
    auto res = DomainBase::New(t, GetMesh(), GetModel(k));
    SetDomain(k, res);
    return res;
};
void Scenario::SetDomain(std::string const &k, std::shared_ptr<DomainBase> const &d) { m_pimpl_->m_domains_[k] = d; }

std::shared_ptr<DomainBase> Scenario::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domains_.find(k);
    return (it == m_pimpl_->m_domains_.end()) ? nullptr : it->second;
}
std::shared_ptr<Schedule> Scenario::NewSchedule(const std::shared_ptr<data::DataNode> &cfg) {
    auto res = Schedule::New(cfg, shared_from_this());
    SetSchedule(res);
    return res;
}
std::shared_ptr<Schedule> Scenario::GetSchedule() const { return m_pimpl_->m_schedule_; }
void Scenario::SetSchedule(std::shared_ptr<Schedule> const &s) { m_pimpl_->m_schedule_ = s; }

void Scenario::Pop(const std::shared_ptr<Patch> &p) { GetMesh()->Pop(p); };
void Scenario::Push(const std::shared_ptr<Patch> &p) { GetMesh()->Push(p); };

void Scenario::InitialCondition(Real time_now) {
    GetMesh()->InitialCondition(time_now);
    for (auto &d : m_pimpl_->m_domains_) { d.second->InitialCondition(time_now); }
}
void Scenario::BoundaryCondition(Real time_now, Real dt) {
    GetMesh()->BoundaryCondition(time_now, dt);
    for (auto &d : m_pimpl_->m_domains_) { d.second->BoundaryCondition(time_now, dt); }
}

void Scenario::ComputeFluxes(Real time_now, Real dt) {
    for (auto &d : m_pimpl_->m_domains_) { d.second->ComputeFluxes(time_now, dt); }
}
Real Scenario::ComputeStableDtOnPatch(Real time_now, Real time_dt) {
    for (auto &d : m_pimpl_->m_domains_) { time_dt = d.second->ComputeStableDtOnPatch(time_now, time_dt); }
    return time_dt;
}

void Scenario::Advance(Real time_now, Real dt) {
    GetMesh()->Advance(time_now, dt);
    for (auto &d : m_pimpl_->m_domains_) { d.second->Advance(time_now, dt); }
}

void Scenario::TagRefinementCells(Real time_now) {
    GetMesh()->TagRefinementCells(time_now);
    for (auto &d : m_pimpl_->m_domains_) { d.second->TagRefinementCells(time_now); }
}
void Scenario::Run() { m_pimpl_->m_schedule_->Run(); }
}  //   namespace engine{
}  // namespace simpla{
