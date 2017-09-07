//
// Created by salmon on 17-4-5.
//

#include "simpla/SIMPLA_config.h"

#include "simpla/geometry/Chart.h"
#include "simpla/geometry/GeoObject.h"

#include "Attribute.h"
#include "Domain.h"
#include "Mesh.h"
#include "Model.h"

namespace simpla {
namespace engine {
struct DomainBase::pimpl_s {
    std::shared_ptr<MeshBase> m_mesh_ = nullptr;
    std::shared_ptr<geometry::GeoObject> m_boundary_ = nullptr;
};
DomainBase::DomainBase() : m_pimpl_(new pimpl_s){};
DomainBase::~DomainBase() { delete m_pimpl_; };
std::shared_ptr<data::DataNode> DomainBase::Serialize() const {
    auto tdb = base_type::Serialize();
    tdb->Set("Boundary", m_pimpl_->m_boundary_->Serialize());
    tdb->Set("Mesh", m_pimpl_->m_mesh_->Serialize());
    tdb->Set("Attributes", AttributeGroup::Serialize());
    return tdb;
}
void DomainBase::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    if (m_pimpl_->m_mesh_ == nullptr) {}
    m_pimpl_->m_mesh_->Deserialize(cfg->Get("Mesh"));
    m_pimpl_->m_boundary_ = geometry::GeoObject::New(cfg->Get("Boundary"));
    AttributeGroup::Deserialize(cfg->Get("Attributes"));
};
void DomainBase::SetMesh(std::shared_ptr<MeshBase> const& m) { m_pimpl_->m_mesh_ = m; };
std::shared_ptr<MeshBase> DomainBase::GetMesh() const { return m_pimpl_->m_mesh_; }

void DomainBase::SetBoundary(std::shared_ptr<geometry::GeoObject> const& g) { m_pimpl_->m_boundary_ = g; }
std::shared_ptr<geometry::GeoObject> DomainBase::GetBoundary() const { return m_pimpl_->m_boundary_; }

bool DomainBase::CheckOverlap(const std::shared_ptr<MeshBlock>& blk) const { return false; }
bool DomainBase::Push(std::shared_ptr<engine::MeshBlock> const& blk, std::shared_ptr<data::DataNode> const& data) {
    if (!CheckOverlap(blk)) { return false; }
    GetMesh()->SetBlock(blk);
    base_type::Push(data);
    return true;
}
std::shared_ptr<data::DataNode> DomainBase::Pop() { return base_type::Pop(); }

void DomainBase::DoSetUp() {
    ASSERT(m_pimpl_->m_mesh_ != nullptr);
    base_type::DoSetUp();
}
void DomainBase::DoUpdate() {
    ASSERT(m_pimpl_->m_mesh_ != nullptr);
    m_pimpl_->m_mesh_->Update();
}
void DomainBase::DoTearDown() {
    m_pimpl_->m_mesh_->TearDown();
    m_pimpl_->m_boundary_.reset();
}

void DomainBase::InitialCondition(Real time_now) {
    Update();
    VERBOSE << "Domain [" << GetName() << "] InitialCondition ";
    PreInitialCondition(this, time_now);
    DoInitialCondition(time_now);
    PostInitialCondition(this, time_now);
}
void DomainBase::BoundaryCondition(Real time_now, Real dt) {
    Update();
    VERBOSE << "Domain [" << GetName() << "] BoundaryCondition ";
    PreBoundaryCondition(this, time_now, dt);
    DoBoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
}

void DomainBase::ComputeFluxes(Real time_now, Real dt) {
    Update();
    VERBOSE << "Domain [" << GetName() << "] ComputeFluxes ";
    PreComputeFluxes(this, time_now, dt);
    DoComputeFluxes(time_now, dt);
    PostComputeFluxes(this, time_now, dt);
}
Real DomainBase::ComputeStableDtOnPatch(Real time_now, Real time_dt) const { return time_dt; }

void DomainBase::Advance(Real time_now, Real dt) {
    Update();
//    if (std::get<0>(GetMesh()->CheckOverlap(GetBoundary())) < EPSILON) { return; }
    VERBOSE << "Domain [" << GetName() << "] Advance ";
    PreAdvance(this, time_now, dt);
    DoAdvance(time_now, dt);
    PostAdvance(this, time_now, dt);
}
void DomainBase::TagRefinementCells(Real time_now) {
    Update();
    if (std::get<0>(GetMesh()->CheckOverlap(GetBoundary())) < EPSILON) { return; }
    PreTagRefinementCells(this, time_now);
    GetMesh()->TagRefinementRange(GetMesh()->GetRange(GetName() + "_BOUNDARY_3"));
    DoTagRefinementCells(time_now);
    PostTagRefinementCells(this, time_now);
}

}  // namespace engine{
}  // namespace simpla{