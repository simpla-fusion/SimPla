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

DomainBase::DomainBase(MeshBase* msh, const Model* model) : m_mesh_(msh), m_model_(model) {}

DomainBase::~DomainBase() = default;

std::shared_ptr<data::DataTable> DomainBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    return (p);
}
void DomainBase::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    ASSERT(m_model_ != nullptr);
    m_boundary_ = m_model_->GetObject(cfg->GetValue<std::string>("Boundary", "Boundary"));
    Click();
};

void DomainBase::DoUpdate() {}
void DomainBase::DoTearDown() {}
void DomainBase::DoInitialize() {}
void DomainBase::DoFinalize() {}

void DomainBase::InitialCondition(Real time_now) {
    VERBOSE << "InitialCondition   \t:" << GetName() << std::endl;
    GetMesh()->AddGeoObject(GetName(), GetBoundary());
    PreInitialCondition(this, time_now);
    DoInitialCondition(time_now);
    PostInitialCondition(this, time_now);
}
void DomainBase::BoundaryCondition(Real time_now, Real dt) {
    VERBOSE << "Boundary Condition \t:" << GetName() << std::endl;
    PreBoundaryCondition(this, time_now, dt);
    DoBoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
}
void DomainBase::Advance(Real time_now, Real dt) {
    VERBOSE << "Advance            \t:" << GetName() << std::endl;
    PreAdvance(this, time_now, dt);
    DoAdvance(time_now, dt);
    PostAdvance(this, time_now, dt);
}

}  // namespace engine{
}  // namespace simpla{