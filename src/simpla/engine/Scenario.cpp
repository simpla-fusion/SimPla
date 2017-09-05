//
// Created by salmon on 17-8-20.
//

#include "simpla/SIMPLA_config.h"

#include "simpla/data/DataNode.h"

#include "Atlas.h"
#include "Domain.h"
#include "Mesh.h"
#include "Model.h"
#include "Scenario.h"
namespace simpla {
namespace engine {

struct Scenario::pimpl_s {
    std::shared_ptr<MeshBase> m_mesh_;
    std::shared_ptr<Atlas> m_atlas_;
    std::map<std::string, std::shared_ptr<Model>> m_models_;
    std::map<std::string, std::shared_ptr<DomainBase>> m_domains_;
};

Scenario::Scenario() : m_pimpl_(new pimpl_s) {}
Scenario::~Scenario() { delete m_pimpl_; }

std::shared_ptr<data::DataNode> Scenario::Serialize() const {
    auto cfg = base_type::Serialize();

    if (m_pimpl_->m_mesh_ != nullptr) cfg->Set("Mesh", GetMesh()->Serialize());
    if (m_pimpl_->m_atlas_ != nullptr) cfg->Set("Atlas", GetAtlas()->Serialize());

    if (auto model = cfg->CreateNode("Model")) {
        for (auto const &item : m_pimpl_->m_models_) { model->Set(item.first, item.second->Serialize()); }
    };

    if (auto domain = cfg->CreateNode("Domain")) {
        for (auto const &item : m_pimpl_->m_domains_) { domain->Set(item.first, item.second->Serialize()); }
    }
    return cfg;
}

void Scenario::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    Initialize();
    base_type::Deserialize(cfg);
    SetMesh(MeshBase::New(cfg->Get("Mesh")));
    m_pimpl_->m_atlas_ = Atlas::New(cfg->Get("Atlas"));

    if (auto model = cfg->Get("Model")) {
        model->Foreach([&](std::string key, std::shared_ptr<data::DataNode> node) {
            return AddModel(key, Model::New(node)) != nullptr ? 1 : 0;
        });
    }
    if (auto domain = cfg->Get("Domain")) {
        domain->Foreach([&](std::string key, std::shared_ptr<data::DataNode> node) {
            if (auto p = SetDomain(key, DomainBase::New(m_pimpl_->m_mesh_, GetModel(key)))) { p->Deserialize(node); };
            return 1;
        });
    }

    Click();
}

void Scenario::DoInitialize() {
    base_type::DoInitialize();
    ASSERT(m_pimpl_->m_mesh_ != nullptr);
    m_pimpl_->m_mesh_->Initialize();

    if (m_pimpl_->m_atlas_ == nullptr) { m_pimpl_->m_atlas_ = Atlas::New(); }
    m_pimpl_->m_atlas_->Initialize();
}
void Scenario::DoFinalize() {
    m_pimpl_->m_domains_.clear();
    m_pimpl_->m_models_.clear();
    m_pimpl_->m_atlas_.reset();
    m_pimpl_->m_mesh_.reset();
    base_type::DoFinalize();
}
void Scenario::DoTearDown() {
    for (auto &item : m_pimpl_->m_domains_) { item.second->TearDown(); }
    for (auto &item : m_pimpl_->m_models_) { item.second->TearDown(); }
    m_pimpl_->m_atlas_->Update();
    m_pimpl_->m_mesh_->Update();
    base_type::DoTearDown();
}
void Scenario::DoUpdate() {
    base_type::DoUpdate();
    m_pimpl_->m_mesh_->Update();
    m_pimpl_->m_atlas_->Update();
    for (auto &item : m_pimpl_->m_models_) { item.second->Update(); }
    for (auto &item : m_pimpl_->m_domains_) { item.second->Update(); }
}

std::shared_ptr<Atlas> Scenario::GetAtlas() const { return m_pimpl_->m_atlas_; }

void Scenario::SetMesh(std::shared_ptr<MeshBase> const &m) { m_pimpl_->m_mesh_ = m; }
std::shared_ptr<MeshBase> Scenario::GetMesh() const { return m_pimpl_->m_mesh_; }

std::shared_ptr<Model> Scenario::AddModel(std::string const &k, std::shared_ptr<Model> m) {
    m_pimpl_->m_models_[k] = m;
    return m_pimpl_->m_models_[k];
}
std::shared_ptr<Model> Scenario::GetModel(std::string const &k) const {
    auto it = m_pimpl_->m_models_.find(k);
    return it == m_pimpl_->m_models_.end() ? nullptr : it->second;
}

std::shared_ptr<DomainBase> Scenario::SetDomain(std::string const &k, std::shared_ptr<DomainBase> d) {
    m_pimpl_->m_domains_[k] = d;
    return m_pimpl_->m_domains_[k];
}
std::shared_ptr<DomainBase> Scenario::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domains_.find(k);
    return (it == m_pimpl_->m_domains_.end()) ? nullptr : it->second;
}
std::map<std::string, std::shared_ptr<DomainBase>> &Scenario::GetDomains() { return m_pimpl_->m_domains_; };
std::map<std::string, std::shared_ptr<DomainBase>> const &Scenario::GetDomains() const { return m_pimpl_->m_domains_; }
std::shared_ptr<data::DataNode> Scenario::Pop() { return GetMesh()->Pop(); };
int Scenario::Push(std::shared_ptr<data::DataNode> const &p) { return GetMesh()->Push(p); };

void Scenario::TagRefinementCells(Real time_now) {
    GetMesh()->TagRefinementCells(time_now);
    for (auto &d : m_pimpl_->m_domains_) { d.second->TagRefinementCells(time_now); }
}
}  //   namespace engine{
}  // namespace simpla{
