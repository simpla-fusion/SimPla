//
// Created by salmon on 17-2-22.
//
#include "Patch.h"
#include <map>
#include "MeshBlock.h"

namespace simpla {
namespace engine {
struct Patch::pimpl_s {
    std::shared_ptr<MeshBlock> m_mesh_ = nullptr;
    std::map<id_type, std::shared_ptr<data::DataBlock>> m_data_;
};
Patch::Patch() : m_pimpl_(new pimpl_s) {}
Patch::~Patch() {}

id_type Patch::GetBlockId() const { return m_pimpl_->m_mesh_ != nullptr ? m_pimpl_->m_mesh_->GetGUID() : NULL_ID; }
std::shared_ptr<MeshBlock> const &Patch::GetMeshBlock() const { return m_pimpl_->m_mesh_; }
void Patch::PushMeshBlock(std::shared_ptr<MeshBlock> const &m) { m_pimpl_->m_mesh_ = m; };
std::shared_ptr<MeshBlock> Patch::PopMeshBlock() {
    auto res = m_pimpl_->m_mesh_;
    m_pimpl_->m_mesh_.reset();
    return res;
}

int Patch::PushData(id_type const &id, std::shared_ptr<data::DataBlock> const &d) {
    auto res = m_pimpl_->m_data_.emplace(id, d);
    if (res.first->second == nullptr) { res.first->second = d; }
    return res.first->second != nullptr ? 1 : 0;
}
std::shared_ptr<data::DataBlock> Patch::PopData(id_type const &id) {
    std::shared_ptr<data::DataBlock> res = nullptr;
    auto it = m_pimpl_->m_data_.find(id);
    if (it != m_pimpl_->m_data_.end()) {
        res = it->second;
        m_pimpl_->m_data_.erase(it);
    }
    return res;
}
}  // namespace engine {
}  // namespace simpla {