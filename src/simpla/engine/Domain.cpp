//
// Created by salmon on 17-4-5.
//

#include "Domain.h"
#include <simpla/mesh/Mesh.h>
#include "Attribute.h"
#include "Patch.h"

namespace simpla {
namespace engine {

struct Domain::pimpl_s {
    std::shared_ptr<model::GeoObject> m_geo_object_;
    std::shared_ptr<MeshBase> m_mesh_base_ = nullptr;
    std::shared_ptr<MeshBase> m_mesh_body_ = nullptr;
    std::shared_ptr<MeshBase> m_mesh_boundary_ = nullptr;

    std::string m_domain_geo_prefix_;
};
Domain::Domain(std::shared_ptr<model::GeoObject> g) : m_pimpl_(new pimpl_s) { SetGeoObject(std::move(g)); }
Domain::~Domain() {}

Domain::Domain(Domain const& other) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_mesh_base_ = other.m_pimpl_->m_mesh_base_;
    m_pimpl_->m_mesh_body_ = other.m_pimpl_->m_mesh_body_;
    m_pimpl_->m_mesh_boundary_ = other.m_pimpl_->m_mesh_boundary_;
    m_pimpl_->m_geo_object_ = other.m_pimpl_->m_geo_object_;
}

Domain::Domain(Domain&& other) noexcept : m_pimpl_(other.m_pimpl_.get()) { other.m_pimpl_.reset(); }

void Domain::swap(Domain& other) {
    std::swap(m_pimpl_->m_mesh_base_, other.m_pimpl_->m_mesh_base_);
    std::swap(m_pimpl_->m_mesh_body_, other.m_pimpl_->m_mesh_body_);
    std::swap(m_pimpl_->m_mesh_boundary_, other.m_pimpl_->m_mesh_boundary_);
    std::swap(m_pimpl_->m_geo_object_, other.m_pimpl_->m_geo_object_);
}

std::string Domain::GetDomainPrefix() const { return m_pimpl_->m_domain_geo_prefix_; };

std::shared_ptr<data::DataTable> Domain::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Name", GetName());
    p->SetValue("GeometryObject", m_pimpl_->m_domain_geo_prefix_);
    return (p);
}
void Domain::Deserialize(const std::shared_ptr<data::DataTable>& cfg) {
    Initialize();
    Click();
    SetName(cfg->GetValue<std::string>("Name", "unnamed"));
    m_pimpl_->m_domain_geo_prefix_ = cfg->GetValue<std::string>("GeometryObject", "");
};

void Domain::DoUpdate() {}
void Domain::DoTearDown() {}
void Domain::DoInitialize() {}
void Domain::DoFinalize() {}

MeshBase const* Domain::GetMesh() const { return m_pimpl_->m_mesh_base_.get(); }
MeshBase const* Domain::GetBodyMesh() const {
    return m_pimpl_->m_mesh_body_ != nullptr ? m_pimpl_->m_mesh_body_.get() : m_pimpl_->m_mesh_base_.get();
}
MeshBase const* Domain::GetBoundaryMesh() const { return m_pimpl_->m_mesh_boundary_.get(); }

void Domain::SetGeoObject(std::shared_ptr<model::GeoObject> g) {
    Click();
    m_pimpl_->m_geo_object_ = std::move(g);
}
const model::GeoObject* Domain::GetGeoObject() const { return m_pimpl_->m_geo_object_.get(); }

void Domain::Push(Patch* patch) {
    Click();
    AttributeGroup::Push(patch);
    m_pimpl_->m_mesh_base_->SetBlock(patch->GetBlock());
    Update();
}
void Domain::Pull(Patch* patch) {
    AttributeGroup::Pull(patch);
    patch->SetBlock(m_pimpl_->m_mesh_base_->GetBlock());
    Click();
    TearDown();
}
void Domain::InitialCondition(Real time_now) {
    PreInitialCondition(this, time_now);
    DoInitialCondition(time_now);
    PostInitialCondition(this, time_now);
}
void Domain::BoundaryCondition(Real time_now, Real dt) {
    PreBoundaryCondition(this, time_now, dt);
    DoBoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
}
void Domain::Advance(Real time_now, Real dt) {
    PreAdvance(this, time_now, dt);
    DoAdvance(time_now, dt);
    PostAdvance(this, time_now, dt);
}
void Domain::InitialCondition(Patch* patch, Real time_now) {
    Push(patch);
    InitialCondition(time_now);
    Pull(patch);
}
void Domain::BoundaryCondition(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    BoundaryCondition(time_now, dt);
    Pull(patch);
}
void Domain::Advance(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    Advance(time_now, dt);
    Pull(patch);
}

}  // namespace engine{
}  // namespace simpla{