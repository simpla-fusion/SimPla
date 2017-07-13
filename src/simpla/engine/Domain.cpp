//
// Created by salmon on 17-4-5.
//

#include "Domain.h"
#include <simpla/engine/Model.h>
#include <simpla/geometry/Chart.h>
#include <simpla/geometry/GeoObject.h>
#include "Attribute.h"
#include "Patch.h"
namespace simpla {
namespace engine {

struct DomainBase::pimpl_s {
    std::shared_ptr<engine::Model> m_model_;
    geometry::Chart const* m_chart_;

    MeshBlock m_mesh_block_;
    std::shared_ptr<std::map<std::string, Range<EntityId>>> m_ranges_;
};
DomainBase::DomainBase() : m_pimpl_(new pimpl_s) {}
DomainBase::~DomainBase() {}

DomainBase::DomainBase(DomainBase const& other) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_model_ = other.m_pimpl_->m_model_;
    m_pimpl_->m_chart_ = other.m_pimpl_->m_chart_;
    m_pimpl_->m_mesh_block_ = other.m_pimpl_->m_mesh_block_;
    m_pimpl_->m_ranges_ = other.m_pimpl_->m_ranges_;
}

DomainBase::DomainBase(DomainBase&& other) noexcept : m_pimpl_(std::move(other.m_pimpl_)) {}

void DomainBase::swap(DomainBase& other) {
    std::swap(m_pimpl_->m_model_, other.m_pimpl_->m_model_);
    std::swap(m_pimpl_->m_chart_, other.m_pimpl_->m_chart_);
    std::swap(m_pimpl_->m_mesh_block_, other.m_pimpl_->m_mesh_block_);
    std::swap(m_pimpl_->m_ranges_, other.m_pimpl_->m_ranges_);
}

std::shared_ptr<data::DataTable> DomainBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Model", GetModel()->Serialize());
    return (p);
}
void DomainBase::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    Initialize();
    Click();
    m_pimpl_->m_model_ = engine::Model::Create(cfg->Get("Model"));
};

void DomainBase::DoUpdate() {}
void DomainBase::DoTearDown() {}
void DomainBase::DoInitialize() {}
void DomainBase::DoFinalize() {}

void DomainBase::SetRange(std::string const& k, Range<EntityId> const& r) {
    Click();
    (*m_pimpl_->m_ranges_)[k] = r;
};
Range<EntityId>* DomainBase::GetRange(std::string const& k) { return &(*m_pimpl_->m_ranges_)[(k)]; };
Range<EntityId> const* DomainBase::GetRange(std::string const& k) const {
    Range<EntityId> const* res = nullptr;
    if (m_pimpl_->m_ranges_ != nullptr) {
        auto it = m_pimpl_->m_ranges_->find(k);
        if (it != m_pimpl_->m_ranges_->end()) { res = &it->second; }
    }
    return res;
};

void DomainBase::SetChart(const geometry::Chart* c) {
    Click();
    m_pimpl_->m_chart_ = c;
}
const geometry::Chart* DomainBase::GetChart() const { return m_pimpl_->m_chart_; }

void DomainBase::SetModel(const std::shared_ptr<engine::Model>& g) {
    m_pimpl_->m_model_ = g;
    Click();
}
const engine::Model* DomainBase::GetModel() const { return m_pimpl_->m_model_.get(); }

void DomainBase::SetBlock(const engine::MeshBlock& blk) { m_pimpl_->m_mesh_block_ = blk; };

const engine::MeshBlock& DomainBase::GetBlock() const { return m_pimpl_->m_mesh_block_; }

id_type DomainBase::GetBlockId() const { return m_pimpl_->m_mesh_block_.GetGUID(); }

void DomainBase::Push(Patch* patch) {
    Click();
    SetBlock(patch->GetBlock());
    m_pimpl_->m_ranges_ = patch->GetRanges();
    AttributeGroup::Push(patch);
    Update();
}
void DomainBase::Pull(Patch* patch) {
    AttributeGroup::Pull(patch);
    patch->SetBlock(GetBlock());
    patch->SetRanges(m_pimpl_->m_ranges_);
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