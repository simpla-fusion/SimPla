//
// Created by salmon on 17-4-5.
//

#include "simpla/SIMPLA_config.h"

#include "simpla/geometry/Chart.h"
#include "simpla/geometry/GeoObject.h"

#include "Attribute.h"
#include "Domain.h"
#include "Model.h"
#include "Patch.h"
namespace simpla {
namespace engine {

struct DomainBase::pimpl_s : public PatchDataPack {
    engine::Model m_model_;
    MeshBlock m_mesh_block_;
    std::map<std::string, Range<EntityId>> m_ranges_;
};
DomainBase::DomainBase(const geometry::Chart* c) : m_chart_(c) {}
DomainBase::~DomainBase() {}

DomainBase::DomainBase(DomainBase const& other) : m_pimpl_(other.m_pimpl_) {}

DomainBase::DomainBase(DomainBase&& other) noexcept : m_pimpl_(std::move(other.m_pimpl_)) {}

void DomainBase::swap(DomainBase& other) {
    std::swap(m_chart_, other.m_chart_);
    std::swap(m_pimpl_, other.m_pimpl_);
}

std::shared_ptr<data::DataTable> DomainBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Model", GetModel().Serialize());
    return (p);
}
void DomainBase::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    Initialize();
    Click();
    m_pimpl_->m_model_.Deserialize(cfg->GetTable("Model"));
};

void DomainBase::DoUpdate() {}
void DomainBase::DoTearDown() {}
void DomainBase::DoInitialize() {
    if (m_pimpl_ == nullptr) { m_pimpl_ = std::make_shared<pimpl_s>(); }
}
void DomainBase::DoFinalize() { m_pimpl_.reset(); }

void DomainBase::SetRange(std::string const& k, Range<EntityId> const& r) {
    Update();
    m_pimpl_->m_ranges_[k] = r;
    Click();
};
Range<EntityId>& DomainBase::GetRange(std::string const& k) { return m_pimpl_->m_ranges_[(k)]; };
Range<EntityId> const& DomainBase::GetRange(std::string const& k) const { return m_pimpl_->m_ranges_.at(k); };

const engine::Model& DomainBase::GetModel() const { return m_pimpl_->m_model_; }
engine::Model& DomainBase::GetModel() { return m_pimpl_->m_model_; }

void DomainBase::SetBlock(const engine::MeshBlock& blk) { MeshBlock(blk).swap(m_mesh_block_); };
const engine::MeshBlock& DomainBase::GetBlock() const { return m_mesh_block_; }
id_type DomainBase::GetBlockId() const { return m_mesh_block_.GetGUID(); }

void DomainBase::Push(Patch* patch) {
    Click();
    m_pimpl_ = std::dynamic_pointer_cast<pimpl_s>(patch->GetPack(GetName()));
    Initialize();
    SetBlock(patch->GetMeshBlock());
    AttributeGroup::Push(patch);
}
void DomainBase::Pull(Patch* patch) {
    AttributeGroup::Pull(patch);
    patch->SetMeshBlock(GetBlock());
    patch->SetPack(GetName(), std::dynamic_pointer_cast<PatchDataPack>(m_pimpl_));
    Finalize();
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