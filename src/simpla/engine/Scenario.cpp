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
    std::shared_ptr<Atlas> m_atlas_ = nullptr;
    std::shared_ptr<Model> m_model_ = nullptr;
    std::map<std::string, std::shared_ptr<DomainBase>> m_domains_;
    std::map<id_type, std::shared_ptr<data::DataNode>> m_patches_;
};

Scenario::Scenario() : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_atlas_ = Atlas::New();
    m_pimpl_->m_model_ = Model::New();
}
Scenario::~Scenario() {
    Finalize();
    delete m_pimpl_;
}

std::shared_ptr<data::DataNode> Scenario::Serialize() const {
    auto cfg = base_type::Serialize();

    cfg->Set("Mesh", GetMesh()->Serialize());
    cfg->Set("Atlas", GetAtlas()->Serialize());
    cfg->Set("Model", GetModel()->Serialize());

    if (auto domain = cfg->CreateNode("Domain")) {
        for (auto const &item : m_pimpl_->m_domains_) { domain->Set(item.first, item.second->Serialize()); }
    }
    return cfg;
}

void Scenario::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_pimpl_->m_mesh_->Deserialize(cfg->Get("Mesh"));
    m_pimpl_->m_atlas_->Deserialize(cfg->Get("Atlas"));
    m_pimpl_->m_model_->Deserialize(cfg->Get("Model"));

    if (auto domain = cfg->Get("Domain")) {
        domain->Foreach([&](std::string key, std::shared_ptr<data::DataNode> node) {
            return SetDomain(key, DomainBase::New(node));
        });
    }

    Click();
}

void Scenario::Synchronize() {}
void Scenario::NextStep() {}
void Scenario::Run() {}
bool Scenario::Done() const { return true; }
void Scenario::Dump() const {}
void Scenario::DoInitialize() {}
void Scenario::DoFinalize() {}

void Scenario::DoSetUp() {
    ASSERT(m_pimpl_->m_mesh_ != nullptr);

    m_pimpl_->m_mesh_->SetUp();
    m_pimpl_->m_model_->SetUp();
    m_pimpl_->m_atlas_->SetUp();

    for (auto &item : m_pimpl_->m_domains_) { item.second->SetUp(); }

    base_type::DoSetUp();
}

void Scenario::DoUpdate() {
    m_pimpl_->m_mesh_->Update();
    m_pimpl_->m_atlas_->Update();
    m_pimpl_->m_model_->Update();
    for (auto &item : m_pimpl_->m_domains_) { item.second->Update(); }
    base_type::DoUpdate();
}
void Scenario::DoTearDown() {
    for (auto &item : m_pimpl_->m_domains_) { item.second->TearDown(); }
    m_pimpl_->m_model_->TearDown();
    m_pimpl_->m_atlas_->TearDown();
    m_pimpl_->m_mesh_->TearDown();
    base_type::DoTearDown();
}
std::shared_ptr<Atlas> Scenario::GetAtlas() const { return m_pimpl_->m_atlas_; }

void Scenario::SetMesh(std::shared_ptr<MeshBase> const &m) {
    ASSERT(!isSetUp());
    m_pimpl_->m_mesh_ = m;
}
std::shared_ptr<MeshBase> Scenario::GetMesh() const { return m_pimpl_->m_mesh_; }

size_type Scenario::SetModel(std::shared_ptr<Model> const &m) {
    ASSERT(!isSetUp());
    m_pimpl_->m_model_ = m;
    return 1;
}

std::shared_ptr<Model> Scenario::GetModel(std::string const &k) const {
    return k.empty() ? m_pimpl_->m_model_ : m_pimpl_->m_model_->Get(k);
}

size_type Scenario::SetDomain(std::string const &k, std::shared_ptr<DomainBase> const &d) {
    ASSERT(!isSetUp());
    size_type count = 0;
    if (auto model = GetModel(k)) {
        m_pimpl_->m_domains_[k] = d;
        m_pimpl_->m_domains_[k]->SetName(k);
        m_pimpl_->m_domains_[k]->SetMesh(GetMesh());
        m_pimpl_->m_domains_[k]->SetModel(model);
        count = 1;
    } else {
        WARNING << " Add domain failed! Model [" << k << "] is not defined. ";
    }
    return count;
}
std::shared_ptr<DomainBase> Scenario::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domains_.find(k);
    return (it == m_pimpl_->m_domains_.end()) ? nullptr : it->second;
}
std::map<std::string, std::shared_ptr<DomainBase>> &Scenario::GetDomains() { return m_pimpl_->m_domains_; };
std::map<std::string, std::shared_ptr<DomainBase>> const &Scenario::GetDomains() const { return m_pimpl_->m_domains_; }

void Scenario::TagRefinementCells(Real time_now) {
    GetMesh()->TagRefinementCells(time_now);
    for (auto &d : m_pimpl_->m_domains_) { d.second->TagRefinementCells(time_now); }
}

size_type Scenario::DeletePatch(id_type id) { return m_pimpl_->m_patches_.erase(id); }

id_type Scenario::SetPatch(id_type id, const std::shared_ptr<data::DataNode> &p) {
    auto res = m_pimpl_->m_patches_.emplace(id, p);
    if (!res.second) { res.first->second = p; }
    return res.first->first;
}

std::shared_ptr<data::DataNode> Scenario::GetPatch(id_type id) {
    std::shared_ptr<data::DataNode> res = nullptr;
    auto it = m_pimpl_->m_patches_.find(id);
    if (it != m_pimpl_->m_patches_.end()) { res = it->second; }
    return res != nullptr ? res : data::DataNode::New();
}

std::shared_ptr<data::DataNode> Scenario::GetPatch(id_type id) const {
    std::shared_ptr<data::DataNode> res = nullptr;
    auto it = m_pimpl_->m_patches_.find(id);
    if (it != m_pimpl_->m_patches_.end()) { res = it->second; }
    return res != nullptr ? res : data::DataNode::New();
}
}  //   namespace engine{
}  // namespace simpla{
