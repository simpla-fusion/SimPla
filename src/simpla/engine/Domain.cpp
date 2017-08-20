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
#include "Patch.h"

namespace simpla {
namespace engine {
DomainBase::DomainBase() {}
DomainBase::DomainBase(std::shared_ptr<MeshBase> const& msh, std::shared_ptr<Model> const& model)
    : m_mesh_(msh), m_model_(model) {}
DomainBase::~DomainBase() {}
void DomainBase::Serialize(std::shared_ptr<data::DataNode> const& cfg) const {
    base_type::Serialize(cfg);
    auto tdb = std::dynamic_pointer_cast<data::DataTable>(cfg);
    if (tdb != nullptr) {
        if (GetGeoBody() != nullptr) { GetGeoBody()->Serialize(tdb->Get("Body")); }
    }
}
void DomainBase::Deserialize(const std::shared_ptr<const data::DataNode>& cfg) {
    base_type::Deserialize(cfg);
    auto tdb = std::dynamic_pointer_cast<const data::DataTable>(cfg);
    if (tdb != nullptr) { m_geo_body_ = geometry::GeoObject::New(tdb->Get("Body")); }
    Click();
};

void DomainBase::DoUpdate() {}
void DomainBase::DoTearDown() {}
void DomainBase::DoInitialize() {}
void DomainBase::DoFinalize() {}

void DomainBase::InitialCondition(Real time_now) {
    Update();
    if (std::get<0>(GetMesh()->CheckOverlap(GetGeoBody().get())) < EPSILON) { return; }
    PreInitialCondition(this, time_now);
    DoInitialCondition(time_now);
    PostInitialCondition(this, time_now);
}
void DomainBase::BoundaryCondition(Real time_now, Real dt) {
    Update();
    if (std::get<0>(GetMesh()->CheckOverlap(GetGeoBody().get())) < EPSILON) { return; }
    PreBoundaryCondition(this, time_now, dt);
    DoBoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
}

void DomainBase::ComputeFluxes(Real time_now, Real dt) {
    Update();
    if (std::get<0>(GetMesh()->CheckOverlap(GetGeoBody().get())) < EPSILON) { return; }
    PreComputeFluxes(this, time_now, dt);
    DoComputeFluxes(time_now, dt);
    PostComputeFluxes(this, time_now, dt);
}
Real DomainBase::ComputeStableDtOnPatch(Real time_now, Real time_dt) const { return time_dt; }

void DomainBase::Advance(Real time_now, Real dt) {
    Update();
    if (std::get<0>(GetMesh()->CheckOverlap(GetGeoBody().get())) < EPSILON) { return; }
    PreAdvance(this, time_now, dt);
    DoAdvance(time_now, dt);
    PostAdvance(this, time_now, dt);
}
void DomainBase::TagRefinementCells(Real time_now) {
    Update();
    if (std::get<0>(GetMesh()->CheckOverlap(GetGeoBody().get())) < EPSILON) { return; }
    PreTagRefinementCells(this, time_now);
    GetMesh()->TagRefinementRange(GetMesh()->GetRange(GetName() + "_BOUNDARY_3"));
    DoTagRefinementCells(time_now);
    PostTagRefinementCells(this, time_now);
}

}  // namespace engine{
}  // namespace simpla{