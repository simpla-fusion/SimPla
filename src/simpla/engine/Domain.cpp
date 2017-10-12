//
// Created by salmon on 17-4-5.
//

#include <simpla/geometry/BoxUtilities.h>
#include "simpla/SIMPLA_config.h"

#include "simpla/geometry/Chart.h"
#include "simpla/geometry/GeoObject.h"

#include "Attribute.h"
#include "Domain.h"

namespace simpla {
namespace engine {
struct DomainBase::pimpl_s {
    std::shared_ptr<geometry::Chart> m_chart_ = nullptr;
    std::shared_ptr<geometry::GeoObject> m_boundary_ = nullptr;
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
    if (m_pimpl_->m_chart_ == nullptr) {
        m_pimpl_->m_chart_ = geometry::Chart::New(cfg->Get("Chart"));
    } else {
        m_pimpl_->m_chart_->Deserialize(cfg->Get("Chart"));
    }
    if (cfg != nullptr) {
        auto lo = cfg->GetValue<point_type>("Box/lo", point_type{0, 0, 0});
        auto hi = cfg->GetValue<point_type>("Box/hi", point_type{1, 1, 1});

        nTuple<int, 3> dims = cfg->GetValue("Dimensions", nTuple<int, 3>{1, 1, 1});

        GetChart()->SetOrigin(lo);
        GetChart()->SetScale((hi - lo) / (dims + 1));

        GetChart()->Deserialize(cfg->Get("Chart"));
    }
    AttributeGroup::Deserialize(cfg->Get("Attributes"));
};
int DomainBase::GetNDIMS() const { return GetChart()->GetNDIMS(); }

void DomainBase::SetChart(std::shared_ptr<geometry::Chart> const& c) { m_pimpl_->m_chart_ = c; }
std::shared_ptr<geometry::Chart> DomainBase::GetChart() { return m_pimpl_->m_chart_; }
std::shared_ptr<const geometry::Chart> DomainBase::GetChart() const { return m_pimpl_->m_chart_; }

void DomainBase::SetMeshBlock(std::shared_ptr<const MeshBlock> const& blk) { m_pimpl_->m_mesh_block_ = blk; };
std::shared_ptr<const MeshBlock> DomainBase::GetMeshBlock() const {
    ASSERT(m_pimpl_->m_mesh_block_ != nullptr);
    return m_pimpl_->m_mesh_block_;
}
// void DomainBase::Push(std::shared_ptr<data::DataNode> const& data) { AttributeGroup::Push(data); }
// std::shared_ptr<data::DataNode> DomainBase::Pop() const { return AttributeGroup::Pop(); }
void DomainBase::Push(const std::shared_ptr<Patch>& p) {
    SetMeshBlock(p->GetMeshBlock());
    AttributeGroup::Push(p);
}
std::shared_ptr<Patch> DomainBase::Pop() const {
    auto res = AttributeGroup::Pop();
    res->SetMeshBlock(GetMeshBlock());
    return res;
}

void DomainBase::SetBoundary(std::shared_ptr<geometry::GeoObject> const& g) { m_pimpl_->m_boundary_ = g; }
std::shared_ptr<geometry::GeoObject> DomainBase::GetBoundary() const { return m_pimpl_->m_boundary_; }
std::shared_ptr<geometry::GeoObject> DomainBase::GetBlockBoundingBox() const {
    return m_pimpl_->m_chart_->GetBoundingShape(m_pimpl_->m_mesh_block_->GetIndexBox());
}
box_type DomainBase::GetBlockBox() const {
    auto idx_box = m_pimpl_->m_mesh_block_->GetIndexBox();
    return std::make_tuple(m_pimpl_->m_chart_->local_coordinates(std::get<0>(idx_box)),
                           m_pimpl_->m_chart_->local_coordinates(std::get<1>(idx_box)));
}

int DomainBase::CheckBoundary() const {
    Real ratio = 1.0;
    if (m_pimpl_->m_boundary_ != nullptr) {
        auto b = GetBlockBoundingBox();
        ratio = m_pimpl_->m_boundary_->Intersection(b)->Measure() / b->Measure();
    }
    return ratio < EPSILON ? 1 : (ratio < 1.0 ? 0 : -1);
}
bool DomainBase::isOutOfBoundary() const {
    FIXME;
    return false;
}

bool DomainBase::isOnBoundary() const {
    FIXME;
    return false;
}
bool DomainBase::isFirstTime() const {
    FIXME;
    return false;
}
void DomainBase::DoSetUp() { base_type::DoSetUp(); }
void DomainBase::DoUpdate() { base_type::DoUpdate(); }
void DomainBase::DoTearDown() { base_type::DoTearDown(); }

void DomainBase::InitialCondition(Real time_now) {
    Update();
    VERBOSE << std::setw(30) << "InitialCondition domain :" << GetName();
    PreInitialCondition(this, time_now);
    DoInitialCondition(time_now);
    PostInitialCondition(this, time_now);
}
void DomainBase::BoundaryCondition(Real time_now, Real dt) {
    Update();
    VERBOSE << std::setw(30) << "BoundaryCondition domain :" << GetName();
    PreBoundaryCondition(this, time_now, dt);
    DoBoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
}

void DomainBase::ComputeFluxes(Real time_now, Real dt) {
    Update();
    VERBOSE << std::setw(30) << "ComputeFluxes domain :" << GetName();
    PreComputeFluxes(this, time_now, dt);
    DoComputeFluxes(time_now, dt);
    PostComputeFluxes(this, time_now, dt);
}
Real DomainBase::ComputeStableDtOnPatch(Real time_now, Real time_dt) const { return time_dt; }

void DomainBase::Advance(Real time_now, Real dt) {
    Update();
    //    if (std::get<0>(GetMesh()->CheckOverlap(GetBoundary())) < EPSILON) { return; }
    VERBOSE << std::setw(30) << "Advance domain :" << GetName();
    PreAdvance(this, time_now, dt);
    DoAdvance(time_now, dt);
    PostAdvance(this, time_now, dt);
}
void DomainBase::TagRefinementCells(Real time_now) {
    Update();
    FIXME;
    //    if (std::get<0>(GetBoundary()->CheckOverlap()) < EPSILON) { return; }
    PreTagRefinementCells(this, time_now);
    //    TagRefinementRange(GetRange(GetName() + "_BOUNDARY_3"));
    DoTagRefinementCells(time_now);
    PostTagRefinementCells(this, time_now);
}

}  // namespace engine{
}  // namespace simpla{