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

    int m_box_in_boundary_ = 1;
    bool m_is_first_time_ = false;
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

        m_pimpl_->m_chart_->SetOrigin(lo);
        m_pimpl_->m_chart_->SetScale((hi - lo) / (dims + 1));
        m_pimpl_->m_chart_->Deserialize(cfg->Get("Chart"));
    }
    AttributeGroup::Deserialize(cfg->Get("Attributes"));
};
int DomainBase::GetNDIMS() const { return GetChart()->GetNDIMS(); }

void DomainBase::SetChart(std::shared_ptr<geometry::Chart> const& c) { m_pimpl_->m_chart_ = c; }
std::shared_ptr<const geometry::Chart> DomainBase::GetChart() const { return m_pimpl_->m_chart_; }

void DomainBase::SetBoundary(std::shared_ptr<geometry::GeoObject> const& g) { m_pimpl_->m_boundary_ = g; }
std::shared_ptr<geometry::GeoObject> DomainBase::GetBoundary() const { return m_pimpl_->m_boundary_; }

void DomainBase::SetMeshBlock(std::shared_ptr<const MeshBlock> const& blk) { m_pimpl_->m_mesh_block_ = blk; };
std::shared_ptr<const MeshBlock> DomainBase::GetMeshBlock() const { return m_pimpl_->m_mesh_block_; }

// void DomainBase::Push(std::shared_ptr<data::DataNode> const& data) { AttributeGroup::Push(data); }
// std::shared_ptr<data::DataNode> DomainBase::Pop() const { return AttributeGroup::Pop(); }
void DomainBase::Push(const std::shared_ptr<Patch>& p) {
    SetMeshBlock(p->GetMeshBlock());
    AttributeGroup::Push(p);
    m_pimpl_->m_box_in_boundary_ = CheckBoundary();

    m_pimpl_->m_is_first_time_ = !AttributeGroup::isInitialized();
}
std::shared_ptr<Patch> DomainBase::Pop() const {
    auto res = AttributeGroup::Pop();
    res->SetMeshBlock(GetMeshBlock());
    m_pimpl_->m_is_first_time_ = false;
    m_pimpl_->m_box_in_boundary_ = -1;
    return res;
}

std::shared_ptr<geometry::GeoObject> DomainBase::GetBlockBoundingBox() const {
    return m_pimpl_->m_chart_->GetBoundingShape(m_pimpl_->m_mesh_block_->GetIndexBox());
}
box_type DomainBase::GetBlockBox() const {
    auto idx_box = m_pimpl_->m_mesh_block_->GetIndexBox();
    return std::make_tuple(m_pimpl_->m_chart_->local_coordinates(std::get<0>(idx_box)),
                           m_pimpl_->m_chart_->local_coordinates(std::get<1>(idx_box)));
}
box_type DomainBase::GetBoundingBox() const {
    return m_pimpl_->m_boundary_ != nullptr ? GetBoundary()->GetBoundingBox()
                                            : box_type{{SP_SNaN, SP_SNaN, SP_SNaN}, {SP_SNaN, SP_SNaN, SP_SNaN}};
}

int DomainBase::CheckBoundary() const {
    return (m_pimpl_->m_boundary_ == nullptr ||
            geometry::isOverlapped(m_pimpl_->m_boundary_->GetBoundingBox(), GetBlockBox()))
               ? 1
               : -1;
}

bool DomainBase::isFirstTime() const { return m_pimpl_->m_is_first_time_; }

void DomainBase::DoSetUp() { base_type::DoSetUp(); }
void DomainBase::DoUpdate() { base_type::DoUpdate(); }
void DomainBase::DoTearDown() { base_type::DoTearDown(); }

void DomainBase::InitialCondition(Real time_now) {
    Update();
    if (CheckBoundary() < 0) { return; }

    VERBOSE << " [ " << std::left << std::setw(20) << GetName() << " ] "
            << "Domain::InitialCondition( time_now =" << time_now << ")"
            << " :  " << std::setw(10) << GetMeshBlock()->GetGUID() << GetMeshBlock()->GetIndexBox();
    PreInitialCondition(this, time_now);
    DoInitialCondition(time_now);
    PostInitialCondition(this, time_now);
}
void DomainBase::BoundaryCondition(Real time_now, Real dt) {
    Update();
    if (CheckBoundary() < 0) { return; }
    VERBOSE << " [ " << std::left << std::setw(20) << GetName() << " ] "
            << "Domain::BoundaryCondition( time_now=" << time_now << " , dt=" << dt << ")"
            << " :  " << std::setw(10) << GetMeshBlock()->GetGUID() << GetMeshBlock()->GetIndexBox();
    PreBoundaryCondition(this, time_now, dt);
    DoBoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
}

void DomainBase::ComputeFluxes(Real time_now, Real time_dt) {
    Update();
    if (CheckBoundary() < 0) { return; }
    VERBOSE << " [ " << std::left << std::setw(20) << GetName() << " ] "
            << "Domain::ComputeFluxes(time_now=" << time_now << " , time_dt=" << time_dt << ")"
            << " :  " << std::setw(10) << GetMeshBlock()->GetGUID() << GetMeshBlock()->GetIndexBox();
    PreComputeFluxes(this, time_now, time_dt);
    DoComputeFluxes(time_now, time_dt);
    PostComputeFluxes(this, time_now, time_dt);
}
Real DomainBase::ComputeStableDtOnPatch(Real time_now, Real time_dt) const {
    if (!isModified() || CheckBoundary() < 0) { return time_dt; }
    VERBOSE << " [ " << std::left << std::setw(20) << GetName() << " ] "
            << "Domain::ComputeStableDtOnPatch( time_now=" << time_now << " , time_dt=" << time_dt << ")"
            << " :  " << std::setw(10) << GetMeshBlock()->GetGUID() << GetMeshBlock()->GetIndexBox();
    return time_dt;
}

void DomainBase::Advance(Real time_now, Real time_dt) {
    Update();
    if (CheckBoundary() < 0) { return; }
    VERBOSE << " [ " << std::left << std::setw(20) << GetName() << " ] "
            << "Domain::Advance(time_now=" << time_now << " , dt=" << time_dt << ")"
            << " :  " << std::setw(10) << GetMeshBlock()->GetGUID() << GetMeshBlock()->GetIndexBox();
    PreAdvance(this, time_now, time_dt);
    DoAdvance(time_now, time_dt);
    PostAdvance(this, time_now, time_dt);
}
void DomainBase::TagRefinementCells(Real time_now) {
    Update();
    if (CheckBoundary() < 0) { return; }
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