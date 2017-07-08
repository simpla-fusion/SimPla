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
    std::shared_ptr<geometry::GeoObject> m_geo_object_;
    std::shared_ptr<MeshBase> m_mesh_ = nullptr;
    std::string m_domain_geo_prefix_;
};
Domain::Domain(const std::shared_ptr<MeshBase>& m, const std::shared_ptr<geometry::GeoObject>& g)
    : SPObject(), m_pimpl_(new pimpl_s) {
    m_pimpl_->m_mesh_ = m;
    m_pimpl_->m_geo_object_ = g;
    Click();
}
Domain::~Domain() {}

Domain::Domain(Domain const& other) { UNIMPLEMENTED; }
Domain::Domain(Domain&& other) { UNIMPLEMENTED; }
void Domain::swap(Domain& other) { UNIMPLEMENTED; }

std::string Domain::GetDomainPrefix() const { return m_pimpl_->m_domain_geo_prefix_; };

std::shared_ptr<data::DataTable> Domain::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Name", GetName());
    p->SetValue("GeometryObject", m_pimpl_->m_domain_geo_prefix_);
    return (p);
}
void Domain::Deserialize(const std::shared_ptr<data::DataTable>& cfg) {
    DoInitialize();
    Click();
    SetName(cfg->GetValue<std::string>("Name", "unnamed"));
    m_pimpl_->m_domain_geo_prefix_ = cfg->GetValue<std::string>("GeometryObject", "");
};

void Domain::DoUpdate() {}
void Domain::DoTearDown() {}
void Domain::DoInitialize() {}
void Domain::DoFinalize() {}
MeshBase const* Domain::GetMesh() const { return m_pimpl_->m_mesh_.get(); }
MeshBase* Domain::GetMesh() { return m_pimpl_->m_mesh_.get(); }

void Domain::SetGeoObject(const geometry::GeoObject& g) {
    Click();
    //    m_pimpl_->m_geo_object_ = g;
}

const geometry::GeoObject& Domain::GetGeoObject() const { return *m_pimpl_->m_geo_object_; }

void Domain::Push(Patch* patch) {
    Click();
    AttributeGroup::Push(patch);
    m_pimpl_->m_mesh_->Push(patch);

    DoUpdate();
}
void Domain::Pull(Patch* patch) {
    AttributeGroup::Pull(patch);
    patch->SetBlock(m_pimpl_->m_mesh_->GetBlock());
    Click();
    DoTearDown();
}

void Domain::DoInitialCondition(Patch* patch, Real time_now) {
    Push(patch);
    PreInitialCondition(this, time_now);
    InitialCondition(time_now);
    PostInitialCondition(this, time_now);
    Pull(patch);
}
void Domain::DoBoundaryCondition(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    PreBoundaryCondition(this, time_now, dt);
    BoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
    Pull(patch);
}
void Domain::DoAdvance(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    PreAdvance(this, time_now, dt);
    Advance(time_now, dt);
    PostAdvance(this, time_now, dt);
    Pull(patch);
}

}  // namespace engine{
}  // namespace simpla{