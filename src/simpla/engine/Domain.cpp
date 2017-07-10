//
// Created by salmon on 17-4-5.
//

#include "Domain.h"
#include <simpla/geometry/Chart.h>
#include <simpla/geometry/GeoObject.h>
#include "../../../to_delete/Mesh.h"
#include "Attribute.h"
#include "Patch.h"
namespace simpla {
namespace engine {

struct DomainBase::pimpl_s {
    std::shared_ptr<geometry::GeoObject> m_geo_object_;
    engine::MeshBlock m_mesh_block_;
    std::shared_ptr<geometry::Chart> m_chart_;
    //    std::shared_ptr<MeshBase> m_mesh_base_ = nullptr;
    //    std::shared_ptr<MeshBase> m_mesh_body_ = nullptr;
    //    std::shared_ptr<MeshBase> m_mesh_boundary_ = nullptr;

    std::string m_domain_geo_prefix_;
};
DomainBase::DomainBase() : m_pimpl_(new pimpl_s) {}
DomainBase::~DomainBase() {}

DomainBase::DomainBase(DomainBase const& other) : m_pimpl_(new pimpl_s) {
    //    m_pimpl_->m_mesh_base_ = other.m_pimpl_->m_mesh_base_;
    //    m_pimpl_->m_mesh_body_ = other.m_pimpl_->m_mesh_body_;
    //    m_pimpl_->m_mesh_boundary_ = other.m_pimpl_->m_mesh_boundary_;
    m_pimpl_->m_geo_object_ = other.m_pimpl_->m_geo_object_;
}

DomainBase::DomainBase(DomainBase&& other) noexcept : m_pimpl_(other.m_pimpl_.get()) { other.m_pimpl_.reset(); }

void DomainBase::swap(DomainBase& other) {
    //    std::swap(m_pimpl_->m_mesh_base_, other.m_pimpl_->m_mesh_base_);
    //    std::swap(m_pimpl_->m_mesh_body_, other.m_pimpl_->m_mesh_body_);
    //    std::swap(m_pimpl_->m_mesh_boundary_, other.m_pimpl_->m_mesh_boundary_);
    std::swap(m_pimpl_->m_geo_object_, other.m_pimpl_->m_geo_object_);
}

std::string DomainBase::GetDomainPrefix() const { return m_pimpl_->m_domain_geo_prefix_; };

std::shared_ptr<data::DataTable> DomainBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Name", GetName());
    p->SetValue("GeometryObject", m_pimpl_->m_domain_geo_prefix_);
    return (p);
}
void DomainBase::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    Initialize();
    Click();
    SetName(cfg->GetValue<std::string>("Name", "unnamed"));
    m_pimpl_->m_domain_geo_prefix_ = cfg->GetValue<std::string>("GeometryObject", "");
};

void DomainBase::DoUpdate() {}
void DomainBase::DoTearDown() {}
void DomainBase::DoInitialize() {}
void DomainBase::DoFinalize() {}

// MeshBase const* DomainBase::GetMesh() const { return m_pimpl_->m_mesh_base_.get(); }
// MeshBase const* DomainBase::GetBodyMesh() const {
//    return m_pimpl_->m_mesh_body_ != nullptr ? m_pimpl_->m_mesh_body_.get() : m_pimpl_->m_mesh_base_.get();
//}
// MeshBase const* DomainBase::GetBoundaryMesh() const { return m_pimpl_->m_mesh_boundary_.get(); }

void DomainBase::SetGeoObject(const std::shared_ptr<geometry::GeoObject>& g) {
    Click();
    m_pimpl_->m_geo_object_ = (g);
}
const geometry::GeoObject* DomainBase::GetGeoObject() const { return m_pimpl_->m_geo_object_.get(); }

void DomainBase::SetChart(std::shared_ptr<geometry::Chart> const& c) {
    Click();
    m_pimpl_->m_chart_ = c;
}
const geometry::Chart* DomainBase::GetChart() const { return m_pimpl_->m_chart_.get(); }

void DomainBase::SetBlock(const engine::MeshBlock& blk) { m_pimpl_->m_mesh_block_ = blk; };
const engine::MeshBlock& DomainBase::GetBlock() const { return m_pimpl_->m_mesh_block_; }
id_type DomainBase::GetBlockId() const { return m_pimpl_->m_mesh_block_.GetGUID(); }
void DomainBase::Push(Patch* patch) {
    Click();
    SetBlock(patch->GetBlock());
    AttributeGroup::Push(patch);
    Update();
}
void DomainBase::Pull(Patch* patch) {
    Click();
    AttributeGroup::Pull(patch);
    patch->SetBlock(GetBlock());
    TearDown();
}
void DomainBase::InitialCondition(Real time_now) {
    PreInitialCondition(this, time_now);
    DoInitialCondition(time_now);
    PostInitialCondition(this, time_now);
}
void DomainBase::BoundaryCondition(Real time_now, Real dt) {
    PreBoundaryCondition(this, time_now, dt);
    DoBoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
}
void DomainBase::Advance(Real time_now, Real dt) {
    PreAdvance(this, time_now, dt);
    DoAdvance(time_now, dt);
    PostAdvance(this, time_now, dt);
}
void DomainBase::InitialCondition(Patch* patch, Real time_now) {
    Push(patch);
    InitialCondition(time_now);
    Pull(patch);
}
void DomainBase::BoundaryCondition(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    BoundaryCondition(time_now, dt);
    Pull(patch);
}
void DomainBase::Advance(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    Advance(time_now, dt);
    Pull(patch);
}

}  // namespace engine{
}  // namespace simpla{