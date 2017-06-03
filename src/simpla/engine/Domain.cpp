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
    Patch* m_patch_ = nullptr;
};
Domain::Domain(const std::shared_ptr<MeshBase>& m, const std::shared_ptr<geometry::GeoObject>& g)
    : SPObject(), m_pimpl_(new pimpl_s) {
    m_pimpl_->m_mesh_ = m;
    m_pimpl_->m_geo_object_ = g;
    Click();
}
Domain::~Domain() {}

std::shared_ptr<data::DataTable> Domain::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Name", GetName());
    return p;
}
void Domain::Deserialize(const std::shared_ptr<DataTable>& t) { UNIMPLEMENTED; };

void Domain::Update() {}
void Domain::TearDown() {}
void Domain::Initialize() {}
void Domain::Finalize() {}
MeshBase const* Domain::GetMesh() const { return m_pimpl_->m_mesh_.get(); }
MeshBase* Domain::GetMesh() { return m_pimpl_->m_mesh_.get(); }

void Domain::SetGeoObject(std::shared_ptr<geometry::GeoObject> const& g) {
    Click();
    m_pimpl_->m_geo_object_ = g;
}

std::shared_ptr<geometry::GeoObject> Domain::GetGeoObject() const { return m_pimpl_->m_geo_object_; }

void Domain::Push(Patch* p) {
    Click();
    m_pimpl_->m_patch_ = p;
    m_pimpl_->m_mesh_->Push(p);
    AttributeGroup::Push(p);
    DoUpdate();
}
void Domain::Pop(Patch* p) {
    AttributeGroup::Pop(p);
    m_pimpl_->m_mesh_->Pop(p);
    m_pimpl_->m_patch_ = nullptr;
    Click();
    DoTearDown();
}

void Domain::DoInitialCondition(Patch* patch, Real time_now) {
    Push(patch);
    PreInitialCondition(this, time_now);
    InitialCondition(time_now);
    PostInitialCondition(this, time_now);
    Pop(patch);
}
void Domain::DoBoundaryCondition(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    PreBoundaryCondition(this, time_now, dt);
    BoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
    Pop(patch);
}
void Domain::DoAdvance(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    PreAdvance(this, time_now, dt);
    Advance(time_now, dt);
    PostAdvance(this, time_now, dt);
    Pop(patch);
}

}  // namespace engine{
}  // namespace simpla{