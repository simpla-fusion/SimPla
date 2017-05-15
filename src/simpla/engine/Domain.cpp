//
// Created by salmon on 17-4-5.
//

#include "Domain.h"
#include "Attribute.h"
#include "MeshBase.h"
#include "Patch.h"

namespace simpla {
namespace engine {

struct Domain::pimpl_s {
    std::map<std::string, std::shared_ptr<geometry::GeoObject>> m_geo_object_;
    std::shared_ptr<geometry::GeoObject> m_geo_shape_;
    std::shared_ptr<Patch> m_patch_ = nullptr;
};
Domain::Domain(std::shared_ptr<geometry::GeoObject> const& g) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_geo_shape_ = g;
    m_pimpl_->m_patch_ = std::make_shared<Patch>();
}
Domain::~Domain() {}

std::shared_ptr<data::DataTable> Domain::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    return p;
}
void Domain::Deserialize(const std::shared_ptr<DataTable>& t) { UNIMPLEMENTED; };

void Domain::SetUp() { GetMesh()->SetUp(); }
void Domain::TearDown() { GetMesh()->TearDown(); }
void Domain::Initialize() { GetMesh()->Initialize(); }
void Domain::Finalize() { GetMesh()->Finalize(); }

void Domain::AddGeoObject(std::string const& k, std::shared_ptr<geometry::GeoObject> const& g) {
    Click();
    m_pimpl_->m_geo_object_[k] = g;
}

std::shared_ptr<geometry::GeoObject> Domain::GetGeoObject(std::string const& k) const {
    std::shared_ptr<geometry::GeoObject> res = nullptr;
    if (k == "") {
        res = m_pimpl_->m_geo_shape_;
    } else {
        auto it = m_pimpl_->m_geo_object_.find(k);
        if (it == m_pimpl_->m_geo_object_.end()) { res = it->second; }
    };
    return res;
}

EntityRange Domain::GetRange(std::string const& k) const {
    auto it = m_pimpl_->m_patch_->m_ranges.find(k);
    return (it != m_pimpl_->m_patch_->m_ranges.end()) ? it->second : EntityRange();
};

EntityRange Domain::GetBodyRange(int IFORM, std::string const& k) const {
    return GetRange(k + "_" + std::to_string(IFORM) + "_BODY");
};
EntityRange Domain::GetBoundaryRange(int IFORM, std::string const& k, bool is_parallel) const {
    return (IFORM == VERTEX || IFORM == VOLUME)
               ? GetRange(k + "_" + std::to_string(IFORM) + "_BOUNDARY")
               : GetRange(k + "_" + std::to_string(IFORM) + (is_parallel ? "_PARA" : "_PERP") + "_BOUNDARY");
};
EntityRange Domain::GetParallelBoundaryRange(int IFORM, std::string const& k) const {
    return GetBoundaryRange(IFORM, k, true);
}
EntityRange Domain::GetPerpendicularBoundaryRange(int IFORM, std::string const& k) const {
    return GetBoundaryRange(IFORM, k, false);
}
EntityRange Domain::GetGhostRange(int IFORM) const { return GetRange("_" + std::to_string(IFORM) + "_GHOST"); }

void Domain::Push(const std::shared_ptr<Patch>& p) {
    Click();
    m_pimpl_->m_patch_ = p;
    GetMesh()->SetBlock(m_pimpl_->m_patch_->GetBlock());
    for (auto& item : GetAllAttributes()) {
        auto k = "_" + std::to_string(item.second->GetIFORM()) + "_BODY";
        auto it = m_pimpl_->m_patch_->m_ranges.find(k);
        EntityRange const* r = (it == m_pimpl_->m_patch_->m_ranges.end()) ? nullptr : &it->second;
        item.second->Push(m_pimpl_->m_patch_->Pop(item.second->GetID()), r);
    }
    DoSetUp();
}
std::shared_ptr<Patch> Domain::PopPatch() {
    m_pimpl_->m_patch_->SetBlock(GetMesh()->GetBlock());
    for (auto& item : GetAllAttributes()) { m_pimpl_->m_patch_->Push(item.second->GetID(), item.second->Pop()); }
    auto res = m_pimpl_->m_patch_;
    m_pimpl_->m_patch_ = nullptr;
    Click();
    DoTearDown();
    return res;
}

std::shared_ptr<Patch> Domain::ApplyInitialCondition(const std::shared_ptr<Patch>& patch, Real time_now) {
    Push(patch);

    if (GetMesh() != nullptr) { GetMesh()->InitializeData(time_now); }
    for (auto const& item : m_pimpl_->m_geo_object_) {
        GetMesh()->RegisterRanges(m_pimpl_->m_patch_->m_ranges, item.second, item.first);
    }
    InitialCondition(time_now);
    OnInitialCondition(this, time_now);

    return PopPatch();
}
std::shared_ptr<Patch> Domain::ApplyBoundaryCondition(const std::shared_ptr<Patch>& patch, Real time_now, Real dt) {
    Push(patch);
    BoundaryCondition(time_now, dt);
    OnBoundaryCondition(this, time_now, dt);
    return PopPatch();
}
std::shared_ptr<Patch> Domain::DoAdvance(const std::shared_ptr<Patch>& patch, Real time_now, Real dt) {
    Push(patch);
    Advance(time_now, dt);
    OnAdvance(this, time_now, dt);
    return PopPatch();
}

}  // namespace engine{
}  // namespace simpla{