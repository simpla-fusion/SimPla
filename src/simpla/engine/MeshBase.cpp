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
    Range<EntityId> m_ranges_[4];
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

Range<EntityId> &MeshBase::GetRange(int iform) { return m_pimpl_->m_ranges_[iform]; };
Range<EntityId> const &MeshBase::GetRange(int iform) const { return m_pimpl_->m_ranges_[iform]; };

void MeshBase::Push(Patch *p) {
    ASSERT(p != nullptr);
    m_pimpl_->m_mesh_block_ = p->GetBlock();
    if (GetGeoObject() != nullptr) {
        m_pimpl_->m_ranges_[VERTEX] = p->GetRange(VERTEX, GetGeoObject()->GetGUID());
        m_pimpl_->m_ranges_[EDGE] = p->GetRange(EDGE, GetGeoObject()->GetGUID());
        m_pimpl_->m_ranges_[FACE] = p->GetRange(FACE, GetGeoObject()->GetGUID());
        m_pimpl_->m_ranges_[VOLUME] = p->GetRange(VOLUME, GetGeoObject()->GetGUID());
    }
    AttributeGroup::Push(p);
}
void MeshBase::Pop(Patch *p) {
    ASSERT(p != nullptr);
    AttributeGroup::Pop(p);
    p->SetBlock(m_pimpl_->m_mesh_block_);
    if (GetGeoObject() != nullptr) {
        p->SetRange(m_pimpl_->m_ranges_[VERTEX], VERTEX, GetGeoObject()->GetGUID());
        p->SetRange(m_pimpl_->m_ranges_[EDGE], EDGE, GetGeoObject()->GetGUID());
        p->SetRange(m_pimpl_->m_ranges_[FACE], FACE, GetGeoObject()->GetGUID());
        p->SetRange(m_pimpl_->m_ranges_[VOLUME], VOLUME, GetGeoObject()->GetGUID());
    }
}
void BoundaryMeshBase::SetUp() {}
}  // {namespace engine
}  // namespace simpla
