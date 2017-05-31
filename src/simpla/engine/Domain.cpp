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
    std::shared_ptr<geometry::GeoObject> m_geo_object_;
    std::shared_ptr<Patch> m_patch_ = nullptr;
    std::shared_ptr<MeshBase> m_mesh_ = nullptr;
};
Domain::Domain(const std::shared_ptr<MeshBase>& m, const std::shared_ptr<geometry::GeoObject>& g)
    : SPObject(), m_pimpl_(new pimpl_s) {
    m_pimpl_->m_mesh_ = m;
    m_pimpl_->m_geo_object_ = g;
    Click();
}
Domain::~Domain() {}

std::shared_ptr<data::DataTable> Domain::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Name", GetName());
    return p;
}
void Domain::Deserialize(const std::shared_ptr<DataTable>& t) { UNIMPLEMENTED; };

void Domain::Update() {
    GetMesh()->Update();
    if (m_pimpl_->m_patch_ == nullptr) { m_pimpl_->m_patch_ = std::make_shared<Patch>(); }
}
void Domain::TearDown() { GetMesh()->TearDown(); }
void Domain::Initialize() { GetMesh()->Initialize(); }
void Domain::Finalize() { GetMesh()->Finalize(); }
MeshBase const* Domain::GetMesh() const { return m_pimpl_->m_mesh_.get(); }
MeshBase* Domain::GetMesh() { return m_pimpl_->m_mesh_.get(); }

void Domain::SetGeoObject(std::shared_ptr<geometry::GeoObject> const& g) {
    Click();
    m_pimpl_->m_geo_object_ = g;
}

std::shared_ptr<geometry::GeoObject> Domain::GetGeoObject() const { return m_pimpl_->m_geo_object_; }

EntityRange Domain::GetRange(std::string const& k) const {
    ASSERT(!isModified());
    auto it = m_pimpl_->m_patch_->m_ranges.find(k);
    //    VERBOSE << FILE_LINE_STAMP << "Get Range " << k << std::endl;
    return (it != m_pimpl_->m_patch_->m_ranges.end()) ? it->second
                                                      : EntityRange{std::make_shared<EmptyRangeBase<EntityId>>()};
};

EntityRange Domain::GetBodyRange(int IFORM, std::string const& k) const {
    return GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + "_BODY");
};
EntityRange Domain::GetBoundaryRange(int IFORM, std::string const& k, bool is_parallel) const {
    auto res =
        (IFORM == VERTEX || IFORM == VOLUME)
            ? GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + "_BOUNDARY")
            : GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + (is_parallel ? "_PARA" : "_PERP") + "_BOUNDARY");
    return res;
};
EntityRange Domain::GetParallelBoundaryRange(int IFORM, std::string const& k) const {
    return GetBoundaryRange(IFORM, k, true);
}
EntityRange Domain::GetPerpendicularBoundaryRange(int IFORM, std::string const& k) const {
    return GetBoundaryRange(IFORM, k, false);
}
EntityRange Domain::GetGhostRange(int IFORM) const {
    return GetRange("." + std::string(EntityIFORMName[IFORM]) + "_GHOST");
}

void Domain::Push(const std::shared_ptr<Patch>& p) {
    Click();
    m_pimpl_->m_patch_ = p;
    GetMesh()->SetBlock(m_pimpl_->m_patch_->GetBlock());
    for (auto& item : GetMesh()->GetAllAttributes()) {
        auto k = "." + std::string(EntityIFORMName[item.second->GetIFORM()]) + "_BODY";

        auto it = m_pimpl_->m_patch_->m_ranges.find(k);
        item.second->Push(m_pimpl_->m_patch_->Pop(item.second->GetID()),
                          (it == m_pimpl_->m_patch_->m_ranges.end()) ? EntityRange{} : it->second);
    }

    DoUpdate();
}
std::shared_ptr<Patch> Domain::PopPatch() {
    m_pimpl_->m_patch_->SetBlock(GetMesh()->GetBlock());
    for (auto& item : GetMesh()->GetAllAttributes()) {
        m_pimpl_->m_patch_->Push(item.second->GetID(), item.second->Pop());
    }
    auto res = m_pimpl_->m_patch_;
    m_pimpl_->m_patch_ = nullptr;
    Click();
    DoTearDown();
    return res;
}

std::shared_ptr<Patch> Domain::DoInitialCondition(const std::shared_ptr<Patch>& patch, Real time_now) {
    Push(patch);

    if (GetMesh() != nullptr) {
        GetMesh()->InitializeData(time_now);
        GetMesh()->RegisterRanges(m_pimpl_->m_patch_->m_ranges, m_pimpl_->m_geo_object_, GetName());
    }
    PreInitialCondition(this, time_now);
    InitialCondition(time_now);
    PostInitialCondition(this, time_now);
    return PopPatch();
}
std::shared_ptr<Patch> Domain::DoBoundaryCondition(const std::shared_ptr<Patch>& patch, Real time_now, Real dt) {
    Push(patch);
    GetMesh()->SetBoundaryCondition(time_now, dt);
    PreBoundaryCondition(this, time_now, dt);
    BoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
    return PopPatch();
}
std::shared_ptr<Patch> Domain::DoAdvance(const std::shared_ptr<Patch>& patch, Real time_now, Real dt) {
    Push(patch);
    PreAdvance(this, time_now, dt);
    Advance(time_now, dt);
    PostAdvance(this, time_now, dt);
    return PopPatch();
}

}  // namespace engine{
}  // namespace simpla{