//
// Created by salmon on 17-4-5.
//

#include "Domain.h"
#include "Attribute.h"
#include "MeshBase.h"
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

data::DataTable Domain::Serialize() const {
    data::DataTable p;
    p.SetValue("Type", GetRegisterName());
    p.SetValue("Name", GetName());
    p.SetValue("GeometryObject", m_pimpl_->m_domain_geo_prefix_);
    return std::move(p);
}
void Domain::Deserialize(const data::DataTable& cfg) {
    DoInitialize();
    Click();
    SetName(cfg.GetValue<std::string>("Name", "unnamed"));
    m_pimpl_->m_domain_geo_prefix_ = cfg.GetValue<std::string>("GeometryObject", "");
};

void Domain::Update() {}
void Domain::TearDown() {}
void Domain::Initialize() {}
void Domain::Finalize() {}
MeshBase const* Domain::GetMesh() const { return m_pimpl_->m_mesh_.get(); }
MeshBase* Domain::GetMesh() { return m_pimpl_->m_mesh_.get(); }

void Domain::SetGeoObject(const geometry::GeoObject& g) {
    Click();
    m_pimpl_->m_geo_object_ = g;
}

const geometry::GeoObject& Domain::GetGeoObject() const { return m_pimpl_->m_geo_object_; }

void Domain::Unpack(Patch&& p) {
    Click();
    AttributeGroup::Unpack(std::forward<Patch>(p));
    m_pimpl_->m_mesh_->Unpack(std::forward<Patch>(p));

    DoUpdate();
}
Patch Domain::Pack() {
    Patch p = AttributeGroup::Pack();
    p.SetBlock(m_pimpl_->m_mesh_->GetBlock());
    Click();
    DoTearDown();
    return std::move(p);
}

Patch Domain::DoInitialCondition(Patch&& patch, Real time_now) {
    Unpack(std::forward<Patch>(patch));
    PreInitialCondition(this, time_now);
    InitialCondition(time_now);
    PostInitialCondition(this, time_now);
    return Pack();
}
Patch Domain::DoBoundaryCondition(Patch&& patch, Real time_now, Real dt) {
    Unpack(std::forward<Patch>(patch));
    PreBoundaryCondition(this, time_now, dt);
    BoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
    return Pack();
}
Patch Domain::DoAdvance(Patch&& patch, Real time_now, Real dt) {
    Unpack(std::forward<Patch>(patch));
    PreAdvance(this, time_now, dt);
    Advance(time_now, dt);
    PostAdvance(this, time_now, dt);
    return Pack();
}

}  // namespace engine{
}  // namespace simpla{