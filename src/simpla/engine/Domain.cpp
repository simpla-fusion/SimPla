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

DomainBase::DomainBase(MeshBase* msh, std::shared_ptr<Model> const& model) : m_mesh_(msh), m_model_(model) {}

DomainBase::~DomainBase() = default;

std::shared_ptr<data::DataTable> DomainBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    if (GetBoundary() != nullptr) { p->SetValue("Boundary", GetBoundary()->Serialize()); }
    return (p);
}
void DomainBase::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    if (cfg->isTable("Boundary")) {
        m_boundary_ = geometry::GeoObject::Create(cfg->Get("Boundary"));
    } else if (m_model_ != nullptr) {
        m_boundary_ = m_model_->GetGeoObject(cfg->GetValue<std::string>("Boundary", ""));
    }

    Click();
};

void DomainBase::DoUpdate() {}
void DomainBase::DoTearDown() {}
void DomainBase::DoInitialize() {}
void DomainBase::DoFinalize() {}

void DomainBase::InitialCondition(Real time_now) {
    Update();
    if (GetBoundary() != nullptr && GetBoundary()->CheckOverlap(GetMesh()->GetBox(0)) < EPSILON) { return; }

    GetMesh()->AddEmbeddedBoundary(GetName(), GetBoundary());
    PreInitialCondition(this, time_now);
    DoInitialCondition(time_now);
    PostInitialCondition(this, time_now);
}
void DomainBase::BoundaryCondition(Real time_now, Real dt) {
    Update();
    PreBoundaryCondition(this, time_now, dt);
    DoBoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
}
void DomainBase::Advance(Real time_now, Real dt) {
    Update();
    PreAdvance(this, time_now, dt);
    DoAdvance(time_now, dt);
    PostAdvance(this, time_now, dt);
}
void DomainBase::TagRefinementCells(Real time_now) {
    Update();
    PreTagRefinementCells(this, time_now);
    DoTagRefinementCells(time_now);
    PostTagRefinementCells(this, time_now);
}

}  // namespace engine{
}  // namespace simpla{