//
// Created by salmon on 17-8-20.
//

#include <simpla/geometry/BoxUtilities.h>
#include <simpla/utilities/type_cast.h>
#include "simpla/SIMPLA_config.h"

#include "simpla/data/DataNode.h"

#include "Atlas.h"
#include "Domain.h"
#include "Scenario.h"
namespace simpla {
namespace engine {

struct Scenario::pimpl_s {
    std::shared_ptr<Atlas> m_atlas_ = nullptr;
    std::map<std::string, std::shared_ptr<DomainBase>> m_domains_;
    std::map<id_type, std::shared_ptr<data::DataNode>> m_patches_;
    std::map<std::string, std::shared_ptr<Attribute>> m_attributes_;
    std::map<std::string, Range<EntityId>> m_ranges_;

    size_type m_step_counter_ = 0;
};

Scenario::Scenario() : m_pimpl_(new pimpl_s) { m_pimpl_->m_atlas_ = Atlas::New(); }
Scenario::~Scenario() {
    Finalize();
    delete m_pimpl_;
}

std::shared_ptr<data::DataNode> Scenario::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Name", GetName());
    res->SetValue("Time", GetTimeNow());

    res->Set("Atlas", GetAtlas()->Serialize());
    res->Set("Chart", m_pimpl_->m_atlas_->GetChart()->Serialize());

    auto domain = data::DataNode::New(data::DataNode::DN_TABLE);
    for (auto const &item : m_pimpl_->m_domains_) { domain->Set(item.first, item.second->Serialize()); }
    res->Set("Domain", domain);

    auto patches = data::DataNode::New(data::DataNode::DN_TABLE);
    for (auto const &item : m_pimpl_->m_patches_) { patches->Set(item.first, item.second); }
    res->Set("Patch", patches);

    return res;
}

void Scenario::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_pimpl_->m_atlas_->Deserialize(cfg->Get("Atlas"));

    if (auto domain = cfg->Get("Domain")) {
        domain->Foreach([&](std::string key, std::shared_ptr<data::DataNode> node) {
            return SetDomain(key, DomainBase::New(node));
        });
    }
    if (auto patches = cfg->Get("Patch")) {
        patches->Foreach([&](std::string key, std::shared_ptr<data::DataNode> node) {
            auto res = m_pimpl_->m_patches_.emplace(static_cast<id_type>(std::stol(key)), node);
            return res.second ? 1 : 0;
        });
    }
    Click();
}

void Scenario::CheckPoint() const {
    std::ostringstream os;
    os << db()->GetValue<std::string>("CheckPointFilePrefix", GetName()) << std::setfill('0') << std::setw(8)
       << GetStepNumber() << "." << db()->GetValue<std::string>("CheckPointFileSuffix", "xmf");
    VERBOSE << std::setw(20) << "Check Point : " << os.str();

    auto dump = data::DataNode::New(os.str());
    dump->Set("Chart", m_pimpl_->m_atlas_->GetChart()->Serialize());
    auto patches = data::DataNode::New(data::DataNode::DN_TABLE);
    auto step = GetStepNumber();
    for (auto const &item : m_pimpl_->m_patches_) {
        patches->Set(item.first, item.second);
        //        if ((step % item.second->GetValue<size_type>("CheckPoint", std::numeric_limits<size_type>::max()) ==
        //        0)) {
        //        }
    }
    dump->Set("Patch", patches);
    dump->Flush();
}

void Scenario::Dump() const {
    std::ostringstream os;

    os << db()->GetValue<std::string>("DumpFilePrefix", GetName()) << "_dump_" << std::setfill('0') << std::setw(8)
       << GetStepNumber() << "." << db()->GetValue<std::string>("DumpFileSuffix", ".h5");
    VERBOSE << std::setw(20) << "Dump : " << os.str();

    auto dump = data::DataNode::New(os.str());
    dump->Set(Serialize());
    dump->Flush();
}
std::map<std::string, std::shared_ptr<Attribute>> &Scenario::GetAttributes() const { return m_pimpl_->m_attributes_; };
Range<EntityId> &Scenario::GetRange(std::string const &k) {
    auto res = m_pimpl_->m_ranges_.emplace(k, Range<EntityId>{});
    return res.first->second;
}
Range<EntityId> const &Scenario::GetRange(std::string const &k) const { return m_pimpl_->m_ranges_.at(k); }

void Scenario::Synchronize() {
#ifdef MPI_FOUND
    FIXME;
#endif
}
void Scenario::NextStep() { ++m_pimpl_->m_step_counter_; }
void Scenario::SetStepNumber(size_type s) { m_pimpl_->m_step_counter_ = s; }
size_type Scenario::GetStepNumber() const { return m_pimpl_->m_step_counter_; }
void Scenario::Run() {}
bool Scenario::Done() const { return true; }

void Scenario::DoInitialize() {}
void Scenario::DoFinalize() {}

void Scenario::DoSetUp() {
    box_type bounding_box;
    auto it = m_pimpl_->m_domains_.begin();
    if (it == m_pimpl_->m_domains_.end() || it->second == nullptr) { return; }
    bounding_box = it->second->GetBoundary()->GetBoundingBox();
    ++it;
    for (; it != m_pimpl_->m_domains_.end(); ++it) {
        if (it->second != nullptr) {
            bounding_box = geometry::Union(bounding_box, it->second->GetBoundary()->GetBoundingBox());
        }
    }

    m_pimpl_->m_atlas_->SetBoundingBox(bounding_box);
    m_pimpl_->m_atlas_->SetUp();
    for (auto &item : m_pimpl_->m_domains_) { item.second->SetUp(); }
    base_type::DoSetUp();
}

void Scenario::DoUpdate() {
    m_pimpl_->m_atlas_->Update();
    base_type::DoUpdate();
}
void Scenario::DoTearDown() {
    for (auto &item : m_pimpl_->m_domains_) { item.second->TearDown(); }
    m_pimpl_->m_atlas_->TearDown();
    base_type::DoTearDown();
}
std::shared_ptr<Atlas> Scenario::GetAtlas() const { return m_pimpl_->m_atlas_; }

size_type Scenario::SetDomain(std::string const &k, std::shared_ptr<DomainBase> const &d) {
    ASSERT(!isSetUp());
    m_pimpl_->m_domains_[k] = d;
    m_pimpl_->m_domains_[k]->SetName(k);
    m_pimpl_->m_domains_[k]->SetChart(m_pimpl_->m_atlas_->GetChart());
    //        m_pimpl_->m_domains_[k]->SetBoundary(g);
    return 1;
}
std::shared_ptr<DomainBase> Scenario::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domains_.find(k);
    return (it == m_pimpl_->m_domains_.end()) ? nullptr : it->second;
}
std::map<std::string, std::shared_ptr<DomainBase>> &Scenario::GetDomains() { return m_pimpl_->m_domains_; };
std::map<std::string, std::shared_ptr<DomainBase>> const &Scenario::GetDomains() const { return m_pimpl_->m_domains_; }

void Scenario::TagRefinementCells(Real time_now) {
    //    GetMesh()->TagRefinementCells(time_now);
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
    return res;
}

std::shared_ptr<data::DataNode> Scenario::GetPatch(id_type id) const {
    std::shared_ptr<data::DataNode> res = nullptr;
    auto it = m_pimpl_->m_patches_.find(id);
    if (it != m_pimpl_->m_patches_.end()) { res = it->second; }
    return res;
}
}  //   namespace engine{
}  // namespace simpla{
