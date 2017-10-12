//
// Created by salmon on 17-8-20.
//
#include "simpla/SIMPLA_config.h"

#include <simpla/data/DataNode.h>
#include <simpla/geometry/BoxUtilities.h>

#include <simpla/parallel/MPIComm.h>
#include <simpla/parallel/Parallel.h>
#include <simpla/utilities/type_cast.h>
#include <fstream>

#include "Atlas.h"
#include "Domain.h"
#include "Scenario.h"
namespace simpla {
namespace engine {

struct Scenario::pimpl_s {
    std::shared_ptr<Atlas> m_atlas_ = nullptr;
    std::map<std::string, std::shared_ptr<Attribute>> m_attrs_;
    std::map<std::string, std::shared_ptr<DomainBase>> m_domains_;

    size_type m_step_counter_ = 0;
    //    void LocalSync(int level = 0);
    //    void MPISync(std::string const &key, std::shared_ptr<Attribute> const &attr, int level = 0);
};

Scenario::Scenario() : m_pimpl_(new pimpl_s) { m_pimpl_->m_atlas_ = Atlas::New(); }
Scenario::~Scenario() {
    Finalize();
    delete m_pimpl_;
}

std::shared_ptr<data::DataNode> Scenario::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Name", GetName());
    res->SetValue("Time", GetTime());

    res->Set("Atlas", GetAtlas()->Serialize());

    auto attributes = data::DataNode::New(data::DataNode::DN_TABLE);
    for (auto const &item : m_pimpl_->m_attrs_) { attributes->Set(item.first, item.second->Serialize()); }
    res->Set("Attributes", attributes);

    auto domain = data::DataNode::New(data::DataNode::DN_TABLE);
    for (auto const &item : m_pimpl_->m_domains_) { domain->Set(item.first, item.second->Serialize()); }
    res->Set("Domains", domain);

    //    auto patches = data::DataNode::New(data::DataNode::DN_TABLE);
    //    for (auto const &item : m_pimpl_->m_patches_) { patches->Set(item.first, item.second->Serialize()); }
    //    res->Set("Patches", patches);

    return res;
}

void Scenario::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_pimpl_->m_atlas_->Deserialize(cfg->Get("Atlas"));

    if (auto domain = cfg->Get("Domains")) {
        domain->Foreach(
            [&](std::string key, std::shared_ptr<data::DataNode> node) { SetDomain(key, DomainBase::New(node)); });
    }
    //    if (auto patches = cfg->Get("Patches")) {
    //        patches->Foreach([&](std::string key, std::shared_ptr<data::DataNode> node) {
    //            m_pimpl_->m_patches_.emplace(static_cast<id_type>(std::stol(key)), node);
    //        });
    //    }

    Click();
}

void Scenario::CheckPoint(size_type step_num) const {
    std::ostringstream os;
    os << db()->GetValue<std::string>("CheckPointFilePrefix", GetName()) << std::setfill('0') << std::setw(8)
       << GetStepNumber() << "." << db()->GetValue<std::string>("CheckPointFileSuffix", "xmf");

    auto dump = data::DataNode::New(os.str());
    dump->Set("Atlas", GetAtlas()->Serialize());

    auto patches = data::DataNode::New(data::DataNode::DN_TABLE);
    m_pimpl_->m_atlas_->Foreach([&](auto const &patch) {
        auto d_patch = patches->CreateNode(std::to_string(patch->GetGUID()), data::DataNode::DN_TABLE);
        for (auto const &attr : m_pimpl_->m_attrs_) {
            auto check_point = attr.second->db()->GetValue<size_type>("CheckPoint", 0);
            if (check_point != 0 && step_num % check_point == 0) {
                if (auto data_blk = patch->GetDataBlock(attr.first)) { d_patch->Set(attr.first, data_blk); }
            }
        }

    });
    dump->Set("Patches", patches);
    dump->SetValue<Real>("Time", GetTime());
    dump->Flush();
}

void Scenario::Dump() const {
    std::ostringstream os;

    os << db()->GetValue<std::string>("DumpFilePrefix", GetName()) << "_dump_" << std::setfill('0') << std::setw(8)
       << GetStepNumber() << "." << db()->GetValue<std::string>("DumpFileSuffix", "h5");
    VERBOSE << std::setw(20) << "Dump : " << os.str();

    auto dump = data::DataNode::New(os.str());
    dump->Set(Serialize());
    dump->Flush();
}
// std::map<std::string, std::shared_ptr<data::DataNode>> const &Scenario::GetAttributes() const {
//    return m_pimpl_->m_attrs_;
//};
// std::map<std::string, std::shared_ptr<data::DataNode>> &Scenario::GetAttributes() { return m_pimpl_->m_attrs_; };

std::shared_ptr<Attribute> Scenario::GetAttribute(std::string const &key) {
    std::shared_ptr<Attribute> res = nullptr;
    auto it = m_pimpl_->m_attrs_.find(key);
    if (it != m_pimpl_->m_attrs_.end()) { res = it->second; }  // OUT_OF_RANGE << "Can not find Attribute" << key;
    return res;
}
std::shared_ptr<Attribute> Scenario::GetAttribute(std::string const &key) const {
    std::shared_ptr<Attribute> res = nullptr;
    auto it = m_pimpl_->m_attrs_.find(key);
    if (it != m_pimpl_->m_attrs_.end()) { res = it->second; }  // OUT_OF_RANGE << "Can not find Attribute" << key;
    return res;
}

// Range<EntityId> &Scenario::GetRange(std::string const &k) {
////    auto res = m_pimpl_->m_ranges_.emplace(k, Range<EntityId>{});
////    return res.first->second;
//}
// Range<EntityId> const &Scenario::GetRange(std::string const &k) const { return m_pimpl_->m_ranges_.at(k); }

void Scenario::Synchronize(int level) {
    ASSERT(level == 0)

    m_pimpl_->m_atlas_->SyncLocal(level);

#ifdef MPI_FOUND
    GLOBAL_COMM.barrier();

    if (GLOBAL_COMM.rank() == 0) {
        for (auto &item : m_pimpl_->m_attrs_) {
            if (item.second->db()->Check("LOCAL")) { continue; }
            parallel::bcast_string(item.first);
            m_pimpl_->m_atlas_->SyncGlobal(item.first, item.second->value_type_info(), item.second->GetNumOfSub(),
                                           level);
        };
        parallel::bcast_string("");
    } else {
        while (1) {
            auto key = parallel::bcast_string();
            if (key.empty()) { break; }
            auto attr = m_pimpl_->m_attrs_.find(key);
            if (attr == m_pimpl_->m_attrs_.end() || attr->second->db()->Check("LOCAL")) {
                RUNTIME_ERROR << "Can not sync local/null attribute \"" << key << "\".";
            }
            m_pimpl_->m_atlas_->SyncGlobal(attr->first, attr->second->value_type_info(), attr->second->GetNumOfSub(),
                                           level);
        }
    }
    GLOBAL_COMM.barrier();

#else
    for (auto &item : m_pimpl_->m_attrs_) {
        if (item.second->db()->Check("LOCAL")) { continue; }
        m_pimpl_->m_atlas_->SyncGlobal(item.first, item.second->value_type_info(), item.second->GetNumOfSub(), level);
    };
#endif  // MPI_FOUND
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

    for (auto &item : m_pimpl_->m_domains_) { item.second->SetUp(); }

    auto chart = m_pimpl_->m_atlas_->GetChart();
    if (!m_pimpl_->m_atlas_->hasBoundingBox()) {
        auto it = m_pimpl_->m_domains_.begin();
        if (it == m_pimpl_->m_domains_.end() || it->second == nullptr) { return; }
        bounding_box = it->second->GetBoundary()->GetBoundingBox();
        ++it;
        for (; it != m_pimpl_->m_domains_.end(); ++it) {
            if (it->second != nullptr) {
                bounding_box = geometry::Union(bounding_box, it->second->GetBoundary()->GetBoundingBox());
            }
        }
        m_pimpl_->m_atlas_->SetBoundingBox(chart->GetBoundingBox(bounding_box));
    }
    m_pimpl_->m_atlas_->SetUp();
    base_type::DoSetUp();

    for (auto &item : m_pimpl_->m_domains_) {
        for (auto *attr : item.second->GetAttributes()) {
            auto res = m_pimpl_->m_attrs_.emplace(attr->GetName(), attr->CreateNew());
            ASSERT(res.first->second->CheckType(*attr));
            res.first->second->db()->Set(attr->db());
            attr->db(res.first->second->db());
        }
    }
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

std::shared_ptr<DomainBase> Scenario::SetDomain(std::string const &k, std::shared_ptr<DomainBase> const &d) {
    ASSERT(!isSetUp());
    if (d != nullptr) {
        m_pimpl_->m_domains_[k] = d;
        m_pimpl_->m_domains_[k]->SetName(k);
        m_pimpl_->m_domains_[k]->SetChart(m_pimpl_->m_atlas_->GetChart());
    }
    return d;
}
std::shared_ptr<DomainBase> Scenario::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domains_.find(k);
    return (it == m_pimpl_->m_domains_.end()) ? nullptr : it->second;
}
std::map<std::string, std::shared_ptr<DomainBase>> &Scenario::GetDomains() { return m_pimpl_->m_domains_; };
std::map<std::string, std::shared_ptr<DomainBase>> const &Scenario::GetDomains() const { return m_pimpl_->m_domains_; }

void Scenario::TagRefinementCells(Real time_now) {
    for (auto &d : m_pimpl_->m_domains_) { d.second->TagRefinementCells(time_now); }
}

// size_type Scenario::DeletePatch(id_type id) { return m_pimpl_->m_patches_.erase(id); }
//
// id_type Scenario::SetPatch(id_type id, const std::shared_ptr<Patch> &p) {
//    auto res = m_pimpl_->m_patches_.emplace(id, p);
//    if (!res.second) { res.first->second = p; }
//    return res.first->first;
//}
//
// std::shared_ptr<Patch> Scenario::GetPatch(id_type id) const {
//    std::shared_ptr<Patch> res = nullptr;
//    auto it = m_pimpl_->m_patches_.find(id);
//    if (it != m_pimpl_->m_patches_.end()) { res = it->second; }
//    return res;
//}
}  //   namespace engine{
}  // namespace simpla{
