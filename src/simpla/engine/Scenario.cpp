//
// Created by salmon on 17-8-20.
//
#include "simpla/SIMPLA_config.h"

#include <simpla/data/DataEntry.h>
#include <simpla/geometry/BoxUtilities.h>

#include <simpla/geometry/GeoEngine.h>
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
};

Scenario::Scenario() : m_pimpl_(new pimpl_s) { m_pimpl_->m_atlas_ = Atlas::New(); }
Scenario::~Scenario() {
    Finalize();
    delete m_pimpl_;
}

std::shared_ptr<data::DataEntry> Scenario::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Time", GetTime());
    res->Set("Atlas", GetAtlas()->Serialize());

    auto attributes = res->CreateNode("Attributes", data::DataEntry::DN_TABLE);
    for (auto const &item : m_pimpl_->m_attrs_) { attributes->Set(item.first, item.second->Serialize()); }

    auto domain = res->CreateNode("Domains", data::DataEntry::DN_TABLE);
    for (auto const &item : m_pimpl_->m_domains_) { domain->Set(item.first, item.second->Serialize()); }

    //    auto patches = data::DataEntry::New(data::DataEntry::DN_TABLE);
    //    for (auto const &item : m_pimpl_->m_patches_) { patches->Set(item.first, item.second->Serialize()); }
    //    res->Set("Patches", patches);

    return res;
}

void Scenario::Deserialize(std::shared_ptr<const data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_pimpl_->m_atlas_->Deserialize(cfg->Get("Atlas"));

    if (auto domain = cfg->Get("Domains")) {
        domain->Foreach([&](std::string key, std::shared_ptr<const data::DataEntry> node) {
            SetDomain(key, DomainBase::Create(node));
        });
    }
    //    if (auto patches = cfg->Get("Patches")) {
    //        patches->Foreach([&](std::string key, std::shared_ptr<data::DataEntry> node) {
    //            m_pimpl_->m_patches_.emplace(static_cast<id_type>(std::stol(key)), node);
    //        });
    //    }

    Click();
}

void Scenario::CheckPoint(size_type step_num) const {
    std::ostringstream os;
    os << GetProperty<std::string>("CheckPointFilePrefix", GetName()) << std::setfill('0') << std::setw(8)
       << GetStepNumber() << "." << GetProperty<std::string>("CheckPointFileSuffix", "xdmf");

    auto dump = data::DataEntry::New(os.str());
    //    dump->Set("Atlas", GetAtlas()->Serialize());
    dump->Set("Atlas/Chart", GetAtlas()->GetChart()->Serialize());
    auto patches = dump->CreateNode("Atlas/Patches", data::DataEntry::DN_TABLE);
    m_pimpl_->m_atlas_->Foreach([&](auto const &patch) {
        auto d_patch = patches->CreateNode(std::to_string(patch->GetGUID()), data::DataEntry::DN_TABLE);
        d_patch->Set("MeshBlock", patch->GetMeshBlock()->Serialize());
        auto d_attrs = d_patch->CreateNode("Attributes", data::DataEntry::DN_TABLE);
        for (auto const &attr : m_pimpl_->m_attrs_) {
            auto check_point = attr.second->GetProperty<size_type>("CheckPoint", 0);
            if (check_point != 0 && step_num % check_point == 0) {
                if (auto data_blk = patch->GetDataBlock(attr.first)) { d_attrs->Set(attr.first, data_blk); }
            }
        }

    });
    dump->SetValue<Real>("Time", GetTime());
    dump->Flush();
}

void Scenario::Dump() const {
    auto prefix = GetProperty<std::string>("DumpFilePrefix", GetName());
    auto suffix = GetProperty<std::string>("DumpFileSuffix", "h5");

    auto geo_prefix = GetProperty<std::string>("GeoFilePrefix", GetName());
    auto geo_suffix = GetProperty<std::string>("GeoFileSuffix", "stl");
    GEO_ENGINE->OpenFile(prefix + "." + geo_suffix);
    for (auto const &d : m_pimpl_->m_domains_) { GEO_ENGINE->Save(d.second->GetBoundary(), d.first); }
    GEO_ENGINE->CloseFile();

    //    std::ostringstream os;
    //
    //    os << prefix << "_dump_" << std::setfill('0') << std::setw(8) << GetStepNumber() << "." << suffix;
    //    VERBOSE << std::setw(20) << "Dump : " << os.str();
    //    auto dump = data::DataEntry::New(os.str());
    //    dump->Set(Serialize());
    //    dump->Flush();
}

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

void Scenario::Synchronize(int level) {
    ASSERT(level == 0)

    m_pimpl_->m_atlas_->SyncLocal(level);

    if (GLOBAL_COMM.size() > 1) {
        GLOBAL_COMM.barrier();

        if (GLOBAL_COMM.rank() == 0) {
            for (auto &item : m_pimpl_->m_attrs_) {
                if (item.second->CheckProperty("LOCAL")) { continue; }
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
                if (attr == m_pimpl_->m_attrs_.end() || attr->second->CheckProperty("LOCAL")) {
                    RUNTIME_ERROR << "Can not sync local/null attribute \"" << key << "\".";
                }
                m_pimpl_->m_atlas_->SyncGlobal(attr->first, attr->second->value_type_info(),
                                               attr->second->GetNumOfSub(), level);
            }
        }
        GLOBAL_COMM.barrier();
    } else {
        for (auto &item : m_pimpl_->m_attrs_) {
            if (item.second->CheckProperty("LOCAL")) { continue; }
            m_pimpl_->m_atlas_->SyncGlobal(item.first, item.second->value_type_info(), item.second->GetNumOfSub(),
                                           level);
        };
    }
}
void Scenario::NextStep() { ++m_pimpl_->m_step_counter_; }
void Scenario::SetStepNumber(size_type s) { m_pimpl_->m_step_counter_ = s; }
size_type Scenario::GetStepNumber() const { return m_pimpl_->m_step_counter_; }
void Scenario::Run() {}
bool Scenario::Done() const { return true; }

void Scenario::DoInitialize() {}
void Scenario::DoFinalize() {}
void Scenario::DoSetUp() {
    ASSERT(m_pimpl_->m_atlas_ != nullptr);
    m_pimpl_->m_atlas_->SetUp();
    for (auto &item : m_pimpl_->m_domains_) {
        if (item.second != nullptr) {
            item.second->SetUp();
            for (auto *attr : item.second->GetAttributes()) {
                auto res = m_pimpl_->m_attrs_.emplace(attr->GetName(), attr->CreateNew());
                ASSERT(res.first->second->CheckType(*attr));
                res.first->second->Link(attr);
            }
        }
    }
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
void Scenario::SetAtlas(std::shared_ptr<Atlas> const &a) { m_pimpl_->m_atlas_ = a; }
std::shared_ptr<const Atlas> Scenario::GetAtlas() const { return m_pimpl_->m_atlas_; }
std::shared_ptr<Atlas> Scenario::GetAtlas() { return m_pimpl_->m_atlas_; }
std::shared_ptr<DomainBase> Scenario::SetDomain(std::string const &k, std::shared_ptr<DomainBase> const &d) {
    ASSERT(!isSetUp());
    if (d != nullptr) {
        m_pimpl_->m_domains_[k] = d;
        m_pimpl_->m_domains_[k]->SetName(k);
        m_pimpl_->m_domains_[k]->SetChart(GetAtlas()->GetChart());
    }
    return d;
}
std::shared_ptr<DomainBase> Scenario::GetDomain(std::string const &k) const {
    auto it = m_pimpl_->m_domains_.find(k);
    return (it == m_pimpl_->m_domains_.end()) ? nullptr : it->second;
}
box_type Scenario::FitBoundingBox() const {
    auto it = m_pimpl_->m_domains_.begin();
    box_type bounding_box = it->second->GetBoundary()->GetBoundingBox();

    for (; it != m_pimpl_->m_domains_.end(); ++it) {
        if (it->second != nullptr) {
            bounding_box = geometry::Union(bounding_box, it->second->GetBoundary()->GetBoundingBox());
        }
    }
    return bounding_box;
}
std::shared_ptr<DomainBase> Scenario::NewDomain(std::string const &s_type, std::string const &k,
                                                std::shared_ptr<const geometry::GeoObject> const &g) {
    SetDomain(k, DomainBase::Create(s_type));
    GetDomain(k)->SetBoundary(g);
    return GetDomain(k);
};
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
