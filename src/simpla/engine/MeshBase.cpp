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
    Range<EntityId> m_ranges_[4];
    Domain *m_domain_ = nullptr;
};
MeshBase::MeshBase(Domain *d) : m_pimpl_(new pimpl_s) { m_pimpl_->m_domain_ = d; }
MeshBase::~MeshBase() {}
Domain *MeshBase::GetDomain() const { return m_pimpl_->m_domain_; }
void MeshBase::SetBlock(std::shared_ptr<MeshBlock> m) { m_pimpl_->m_mesh_block_ = m; }
std::shared_ptr<MeshBlock> MeshBase::GetBlock() const { return m_pimpl_->m_mesh_block_; }
id_type MeshBase::GetBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->GetGUID();
}

void MeshBase::SetUp() {}
void MeshBase::TearDown() {}
void MeshBase::Initialize() {}
void MeshBase::Finalize() {}

void MeshBase::InitializeData(Real time_now) {}

std::shared_ptr<data::DataTable> MeshBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    return p;
}
void MeshBase::Deserialize(const std::shared_ptr<DataTable> &) {}

bool MeshBase::isFullCovered() const { return true; }
bool MeshBase::isBoundary() const { return false; }

Range<EntityId> &MeshBase::GetRange(int iform) { return m_pimpl_->m_ranges_[iform]; };
Range<EntityId> const &MeshBase::GetRange(int iform) const { return m_pimpl_->m_ranges_[iform]; };

}  // {namespace engine
}  // namespace simpla
