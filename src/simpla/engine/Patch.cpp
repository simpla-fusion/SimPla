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
    std::map<id_type, std::shared_ptr<data::DataBlock>> m_data_;
};
Patch::Patch(std::shared_ptr<MeshBlock> const &m) : m_pimpl_(new pimpl_s) { m_pimpl_->m_mesh_ = m; }
Patch::~Patch() {}

void Patch::Push(std::shared_ptr<Patch> const &other) const {
    ASSERT(m_pimpl_->m_mesh_ == other->m_pimpl_->m_mesh_);
    m_pimpl_->m_data_.insert(other->m_pimpl_->m_data_.begin(), other->m_pimpl_->m_data_.end());
}

id_type Patch::GetMeshBlockId() const { return GetMeshBlock()->GetGUID(); }

void Patch::SetMeshBlock(std::shared_ptr<MeshBlock> const &m) {
    ASSERT(m_pimpl_->m_mesh_ == nullptr);
    m_pimpl_->m_mesh_ = m;
};
int Patch::SetDataBlock(id_type const &id, std::shared_ptr<data::DataBlock> const &d) {
    auto res = m_pimpl_->m_data_.emplace(id, d);
    if (res.first->second == nullptr) { res.first->second = d; }
    return res.first->second != nullptr ? 1 : 0;
}

std::shared_ptr<MeshBlock> Patch::GetMeshBlock() const { return m_pimpl_->m_mesh_; }
std::shared_ptr<data::DataBlock> Patch::GetDataBlock(id_type const &id) const {
    auto it = m_pimpl_->m_data_.find(id);
    return it == m_pimpl_->m_data_.end() ? nullptr : it->second;
}
std::map<id_type, std::shared_ptr<data::DataBlock>> &Patch::GetAllDataBlock() const { return m_pimpl_->m_data_; };
}  // namespace engine {
}  // namespace simpla {