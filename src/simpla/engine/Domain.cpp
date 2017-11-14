//
// Created by salmon on 17-4-5.
//

#include <simpla/geometry/BoxUtilities.h>
#include <simpla/geometry/GeoAlgorithm.h>
#include "simpla/SIMPLA_config.h"

#include "simpla/geometry/Chart.h"
#include "simpla/geometry/GeoObject.h"

#include "Attribute.h"
#include "Domain.h"

namespace simpla {
namespace engine {
struct DomainBase::pimpl_s {
    std::shared_ptr<const geometry::Chart> m_chart_ = nullptr;
    std::shared_ptr<const geometry::GeoObject> m_boundary_ = nullptr;
    std::shared_ptr<const MeshBlock> m_mesh_block_ = nullptr;
};
DomainBase::DomainBase() : m_pimpl_(new pimpl_s){};
DomainBase::~DomainBase() { delete m_pimpl_; };

std::shared_ptr<data::DataNode> DomainBase::Serialize() const {
    auto tdb = base_type::Serialize();
    this->OnSerialize(this, tdb);
    ASSERT(m_pimpl_->m_chart_ != nullptr);
    tdb->Set("Chart", m_pimpl_->m_chart_->Serialize());
    if (m_pimpl_->m_boundary_ != nullptr) { tdb->Set("Boundary", m_pimpl_->m_boundary_->Serialize()); }
    tdb->Set("Attributes", AttributeGroup::Serialize());
    return tdb;
}
void DomainBase::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    this->OnDeserialize(this, cfg);
    m_pimpl_->m_boundary_ = geometry::GeoObject::New(cfg->Get("Boundary"));
    if (cfg->Get("Chart") != nullptr) { m_pimpl_->m_chart_ = geometry::Chart::New(cfg->Get("Chart")); }
    AttributeGroup::Deserialize(cfg->Get("Attributes"));
};

void DomainBase::SetChart(std::shared_ptr<const geometry::Chart> const& c) { m_pimpl_->m_chart_ = c; }
std::shared_ptr<const geometry::Chart> DomainBase::GetChart() const { return m_pimpl_->m_chart_; }

void DomainBase::SetBoundary(std::shared_ptr<const geometry::GeoObject> const& g) { m_pimpl_->m_boundary_ = g; }
std::shared_ptr<const geometry::GeoObject> DomainBase::GetBoundary() const { return m_pimpl_->m_boundary_; }

void DomainBase::SetMeshBlock(std::shared_ptr<const MeshBlock> const& blk) { m_pimpl_->m_mesh_block_ = blk; };
std::shared_ptr<const MeshBlock> DomainBase::GetMeshBlock() const { return m_pimpl_->m_mesh_block_; }
box_type DomainBase::GetBlockBox() const { return GetChart()->GetBoxUVW(GetMeshBlock()->GetIndexBox()); }
void DomainBase::Push(const std::shared_ptr<Patch>& p) {
    SetMeshBlock(p->GetMeshBlock());
    AttributeGroup::Push(p);
}
std::shared_ptr<Patch> DomainBase::Pop() const {
    auto res = AttributeGroup::Pop();
    res->SetMeshBlock(GetMeshBlock());
    return res;
}

// box_type DomainBase::GetBoundingBox() const {
//    return GetBoundary() != nullptr
//               ? GetBoundary()->GetBoundingBox()
//               : box_type{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}, {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
//}

int DomainBase::CheckBlockInBoundary() const {
    auto blk = GetChart()->GetBoundingShape(GetMeshBlock()->GetIndexBox());
    return (GetBoundary() == nullptr || GetBoundary()->CheckIntersection(blk, SP_GEO_DEFAULT_TOLERANCE)) ? 1 : -1;
}

bool DomainBase::IsInitialized() const { return AttributeGroup::IsInitialized(); }

void DomainBase::DoSetUp() { base_type::DoSetUp(); }
void DomainBase::DoUpdate() { base_type::DoUpdate(); }
void DomainBase::DoTearDown() { base_type::DoTearDown(); }

void DomainBase::InitialCondition(Real time_now) {
    Update();
    if (CheckBlockInBoundary() < 0) { return; }

    VERBOSE << " [ " << std::left << std::setw(20) << GetName() << " ] "
            << "Domain::InitialCondition( time_now =" << time_now << ")"
            << " :  " << std::setw(10) << GetMeshBlock()->GetGUID() << GetMeshBlock()->GetIndexBox();
    PreInitialCondition(this, time_now);
    DoInitialCondition(time_now);
    PostInitialCondition(this, time_now);
}
void DomainBase::BoundaryCondition(Real time_now, Real dt) {
    Update();
    if (CheckBlockInBoundary() < 0) { return; }
    VERBOSE << " [ " << std::left << std::setw(20) << GetName() << " ] "
            << "Domain::BoundaryCondition( time_now=" << time_now << " , dt=" << dt << ")"
            << " :  " << std::setw(10) << GetMeshBlock()->GetGUID() << GetMeshBlock()->GetIndexBox();
    PreBoundaryCondition(this, time_now, dt);
    DoBoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
}

void DomainBase::ComputeFluxes(Real time_now, Real time_dt) {
    Update();
    if (CheckBlockInBoundary() < 0) { return; }
    VERBOSE << " [ " << std::left << std::setw(20) << GetName() << " ] "
            << "Domain::ComputeFluxes(time_now=" << time_now << " , time_dt=" << time_dt << ")"
            << " :  " << std::setw(10) << GetMeshBlock()->GetGUID() << GetMeshBlock()->GetIndexBox();
    PreComputeFluxes(this, time_now, time_dt);
    DoComputeFluxes(time_now, time_dt);
    PostComputeFluxes(this, time_now, time_dt);
}
Real DomainBase::ComputeStableDtOnPatch(Real time_now, Real time_dt) const {
    if (!isModified() || CheckBlockInBoundary() < 0) { return time_dt; }
    VERBOSE << " [ " << std::left << std::setw(20) << GetName() << " ] "
            << "Domain::ComputeStableDtOnPatch( time_now=" << time_now << " , time_dt=" << time_dt << ")"
            << " :  " << std::setw(10) << GetMeshBlock()->GetGUID() << GetMeshBlock()->GetIndexBox();
    return time_dt;
}

void DomainBase::Advance(Real time_now, Real time_dt) {
    Update();
    if (CheckBlockInBoundary() < 0) { return; }
    VERBOSE << " [ " << std::left << std::setw(20) << GetName() << " ] "
            << "Domain::Advance(time_now=" << time_now << " , dt=" << time_dt << ")"
            << " :  " << std::setw(10) << GetMeshBlock()->GetGUID() << GetMeshBlock()->GetIndexBox();
    PreAdvance(this, time_now, time_dt);
    DoAdvance(time_now, time_dt);
    PostAdvance(this, time_now, time_dt);
}
void DomainBase::TagRefinementCells(Real time_now) {
    Update();
    if (CheckBlockInBoundary() < 0) { return; }
    VERBOSE << " [ " << std::left << std::setw(20) << GetName() << " ] "
            << "Domain::TagRefinementCells(time_now=" << time_now << ")"
            << " :  " << std::setw(10) << GetMeshBlock()->GetGUID() << GetMeshBlock()->GetIndexBox();

    PreTagRefinementCells(this, time_now);
    //    TagRefinementRange(GetRange(GetName() + "_BOUNDARY_3"));
    DoTagRefinementCells(time_now);
    PostTagRefinementCells(this, time_now);
}
std::shared_ptr<DomainBase> DomainBase::AddEmbeddedDomain(std::string const& k, std::shared_ptr<DomainBase> const& b) {
    return b;
}

}  // namespace engine{
}  // namespace simpla{