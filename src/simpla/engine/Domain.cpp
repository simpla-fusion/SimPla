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

namespace simpla {
namespace engine {
struct DomainBase::pimpl_s {
    std::shared_ptr<MeshBase> m_mesh_ = nullptr;
    std::shared_ptr<engine::Model> m_model_ = nullptr;
};
DomainBase::DomainBase() : m_pimpl_(new pimpl_s){};
DomainBase::DomainBase(std::shared_ptr<MeshBase> const& msh, std::shared_ptr<Model> const& model) : DomainBase() {
    m_pimpl_->m_mesh_ = (msh);
    m_pimpl_->m_model_ = (model);
}
DomainBase::~DomainBase() { delete m_pimpl_; };
std::shared_ptr<data::DataNode> DomainBase::Serialize() const {
    auto tdb = base_type::Serialize();
    tdb->SetValue("Model", GetModel()->GetName());
    tdb->SetValue("Mesh", GetMesh()->GetName());
    return tdb;
}
void DomainBase::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    //    if (m_mesh_ == nullptr) {
    //        m_mesh_ = MeshBase::New(cfg->Get("Mesh"));
    //    } else {
    //        m_mesh_->Deserialize(cfg->Get("Mesh"));
    //    }
    //    if (m_mesh_ == nullptr) {
    //        m_model_ = Model::New(cfg->Get("Model"));
    //    } else {
    //        m_model_->Deserialize(cfg->Get("Model"));
    //    }
    Click();
};

void DomainBase::SetMesh(std::shared_ptr<MeshBase> const& m) {
    ASSERT(!isSetUp());
    m_pimpl_->m_mesh_ = m;
}
std::shared_ptr<MeshBase> DomainBase::GetMesh() const { return m_pimpl_->m_mesh_; }
void DomainBase::SetModel(std::shared_ptr<Model> const& m) {
    ASSERT(!isSetUp());
    m_pimpl_->m_model_ = m;
}
std::shared_ptr<Model> DomainBase::GetModel() const { return m_pimpl_->m_model_; }

void DomainBase::SetBlock(const std::shared_ptr<MeshBlock>& blk) { GetMesh()->SetBlock(blk); }
std::shared_ptr<MeshBlock> DomainBase::GetBlock() const { return GetMesh()->GetBlock(); }

bool DomainBase::CheckOverlap(const std::shared_ptr<MeshBlock>& blk) const { return false; }
bool DomainBase::Push(std::shared_ptr<engine::MeshBlock> const& blk, std::shared_ptr<data::DataNode> const& data) {
    if (!CheckOverlap(blk)) { return false; }
    SetBlock(blk);
    base_type::Push(data);
    return true;
}
std::shared_ptr<data::DataNode> DomainBase::Pop() { return base_type::Pop(); }

void DomainBase::DoSetUp() {
    ASSERT(m_pimpl_->m_mesh_ != nullptr);
    //    ASSERT(m_pimpl_->m_model_ != nullptr);
    base_type::DoSetUp();
}
void DomainBase::DoUpdate() {
    ASSERT(m_pimpl_->m_mesh_ != nullptr);
    ASSERT(m_pimpl_->m_model_ != nullptr);
    m_pimpl_->m_mesh_->Update();
    m_pimpl_->m_model_->Update();
}
void DomainBase::DoTearDown() {
    m_pimpl_->m_mesh_.reset();
    m_pimpl_->m_model_.reset();
}

void DomainBase::InitialCondition(Real time_now) {
    Update();
    if (std::get<0>(GetMesh()->CheckOverlap(GetModel()->GetBoundary())) < EPSILON) { return; }
    PreInitialCondition(this, time_now);
    DoInitialCondition(time_now);
    PostInitialCondition(this, time_now);
}
void DomainBase::BoundaryCondition(Real time_now, Real dt) {
    Update();
    if (std::get<0>(GetMesh()->CheckOverlap(GetModel()->GetBoundary())) < EPSILON) { return; }
    PreBoundaryCondition(this, time_now, dt);
    DoBoundaryCondition(time_now, dt);
    PostBoundaryCondition(this, time_now, dt);
}

void DomainBase::ComputeFluxes(Real time_now, Real dt) {
    Update();
    if (std::get<0>(GetMesh()->CheckOverlap(GetModel()->GetBoundary())) < EPSILON) { return; }
    PreComputeFluxes(this, time_now, dt);
    DoComputeFluxes(time_now, dt);
    PostComputeFluxes(this, time_now, dt);
}
Real DomainBase::ComputeStableDtOnPatch(Real time_now, Real time_dt) const { return time_dt; }

void DomainBase::Advance(Real time_now, Real dt) {
    Update();
    if (std::get<0>(GetMesh()->CheckOverlap(GetModel()->GetBoundary())) < EPSILON) { return; }
    PreAdvance(this, time_now, dt);
    DoAdvance(time_now, dt);
    PostAdvance(this, time_now, dt);
}
void DomainBase::TagRefinementCells(Real time_now) {
    Update();
    if (std::get<0>(GetMesh()->CheckOverlap(GetModel()->GetBoundary())) < EPSILON) { return; }
    PreTagRefinementCells(this, time_now);
    GetMesh()->TagRefinementRange(GetMesh()->GetRange(GetName() + "_BOUNDARY_3"));
    DoTagRefinementCells(time_now);
    PostTagRefinementCells(this, time_now);
}

}  // namespace engine{
}  // namespace simpla{