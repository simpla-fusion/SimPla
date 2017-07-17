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

// DomainBase::DomainBase(DomainBase const& other) : SPObject(other), m_mesh_(other.m_mesh_), m_model_(other.m_model_)
// {}
//
// DomainBase::DomainBase(DomainBase&& other) noexcept
//    : SPObject(std::move(other)), m_mesh_(other.m_mesh_), m_model_(other.m_model_) {}
//
// void DomainBase::swap(DomainBase& other) {
//    SPObject::swap(other);
//    std::swap(m_model_, other.m_model_);
//    std::swap(m_mesh_, other.m_mesh_);
//    std::swap(m_boundary_, other.m_boundary_);
//}

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

void DomainBase::SetRange(std::string const& k, Range<EntityId> const& r) { GetMesh()->SetRange(k, r); };
Range<EntityId>& DomainBase::GetRange(std::string const& k) { return GetMesh()->GetRange(k); };
Range<EntityId> DomainBase::GetRange(std::string const& k) const { return GetMesh()->GetRange(k); };

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