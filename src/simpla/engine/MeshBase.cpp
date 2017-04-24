//
// Created by salmon on 16-11-24.
//
#include "MeshBase.h"
#include <simpla/geometry/GeoObject.h>
#include <simpla/utilities/EntityId.h>
#include "Attribute.h"
#include "Domain.h"
#include "MeshBlock.h"
#include "Model.h"
#include "Patch.h"
namespace simpla {
namespace engine {

struct MeshBase::pimpl_s {
    std::shared_ptr<MeshBlock> m_mesh_block_;
    std::shared_ptr<geometry::GeoObject> m_geo_obj_;
    std::shared_ptr<Chart> m_chart_;
    std::shared_ptr<Patch> m_patch_;
    std::map<int, Range<EntityId>> m_ranges_;
    Real m_time_ = 0.0;
};
MeshBase::MeshBase(std::shared_ptr<Chart> c) : m_pimpl_(new pimpl_s) { m_pimpl_->m_chart_ = c; }
MeshBase::~MeshBase() {}

Real MeshBase::GetTime() const { return m_pimpl_->m_time_; }
void MeshBase::SetBlock(std::shared_ptr<MeshBlock> m) { m_pimpl_->m_mesh_block_ = m; }
std::shared_ptr<MeshBlock> MeshBase::GetBlock() const { return m_pimpl_->m_mesh_block_; }
id_type MeshBase::GetBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->GetGUID();
}

void MeshBase::SetGeoObject(std::shared_ptr<geometry::GeoObject> g) { m_pimpl_->m_geo_obj_ = g; }
std::shared_ptr<geometry::GeoObject> MeshBase::GetGeoObject() const { return m_pimpl_->m_geo_obj_; }

void MeshBase::SetChart(std::shared_ptr<Chart> c) { m_pimpl_->m_chart_ = c; }
std::shared_ptr<Chart> MeshBase::GetChart() const { return m_pimpl_->m_chart_; }

void MeshBase::SetUp() {}
void MeshBase::TearDown() {}
void MeshBase::Initialize() {}
void MeshBase::Finalize() {}

void MeshBase::InitializeData(Real time_now) { m_pimpl_->m_time_ = time_now; }

std::shared_ptr<data::DataTable> MeshBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetClassName());
    return p;
}
void MeshBase::Deserialize(std::shared_ptr<data::DataTable>) {}
Range<EntityId> MeshBase::GetRange(int iform) const {
    return Range<EntityId>(std::make_shared<ContinueRange<EntityId>>(GetBlock()->GetIndexBox(), iform));
};
void MeshBase::Push(std::shared_ptr<Patch> p) {
    ASSERT(p != nullptr);
    m_pimpl_->m_patch_ = p;
    m_pimpl_->m_mesh_block_ = p->GetBlock();
    AttributeGroup::PushPatch(m_pimpl_->m_patch_);
}
std::shared_ptr<Patch> MeshBase::Pop() {
    AttributeGroup::PopPatch(m_pimpl_->m_patch_);
    m_pimpl_->m_patch_->SetChart(m_pimpl_->m_chart_);
    return m_pimpl_->m_patch_;
}
void BoundaryMeshBase::SetUp() {}
}  // {namespace engine
}  // namespace simpla
