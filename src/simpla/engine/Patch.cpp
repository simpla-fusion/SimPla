//
// Created by salmon on 17-2-22.
//
#include "Patch.h"
#include <map>
#include "MeshBlock.h"

namespace simpla {
namespace engine {
struct Patch::pimpl_s {
    std::shared_ptr<MeshBlock> m_mesh_;
    std::map<id_type, std::shared_ptr<DataBlock> > m_data_;
};
Patch::Patch() : m_pimpl_(new pimpl_s) {}
Patch::~Patch() {}
id_type Patch::GetMeshBlockId() const { return m_pimpl_->m_mesh_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_->id(); }
std::shared_ptr<MeshBlock> const &Patch::GetMeshBlock() const { return m_pimpl_->m_mesh_; }
void Patch::SetMeshBlock(std::shared_ptr<MeshBlock> const &m) { m_pimpl_->m_mesh_ = m; }
void Patch::SetDataBlock(id_type const &id, std::shared_ptr<DataBlock> const &p) { m_pimpl_->m_data_[id] = p; }
std::shared_ptr<DataBlock> const &Patch::GetDataBlock(id_type const &id) const { return m_pimpl_->m_data_.at(id); }
std::shared_ptr<DataBlock> &Patch::GetDataBlock(id_type const &id) {
    return m_pimpl_->m_data_.emplace(id, std::shared_ptr<DataBlock>(nullptr)).first->second;
}
}  // namespace engine {
}  // namespace simpla {