//
// Created by salmon on 17-2-22.
//
#include "Patch.h"
#include <map>
#include "MeshBlock.h"
#include "simpla/geometry/GeoObject.h"

namespace simpla {
namespace engine {
struct Patch::pimpl_s {
    std::shared_ptr<Mesh> m_mesh_ = nullptr;
    std::map<id_type, std::shared_ptr<data::DataBlock>> m_data_;
};
Patch::Patch() : m_pimpl_(new pimpl_s) {}
Patch::~Patch() {}

id_type Patch::GetBlockId() const { return m_pimpl_->m_mesh_ != nullptr ? m_pimpl_->m_mesh_->GetBlockId() : NULL_ID; }

void Patch::PushMesh(std::shared_ptr<Mesh> const &m) { m_pimpl_->m_mesh_ = m; };
std::shared_ptr<Mesh> Patch::PopMesh() const { return m_pimpl_->m_mesh_; }

int Patch::Push(id_type const &id, std::shared_ptr<data::DataBlock> const &d) {
    auto res = m_pimpl_->m_data_.emplace(id, d);
    if (res.first->second == nullptr) { res.first->second = d; }
    return res.first->second != nullptr ? 1 : 0;
}
std::shared_ptr<data::DataBlock> Patch::Pop(id_type const &id) const {
    std::shared_ptr<data::DataBlock> res = nullptr;
    auto it = m_pimpl_->m_data_.find(id);
    if (it != m_pimpl_->m_data_.end()) { res = it->second; }
    return res;
}

}  // namespace engine {
}  // namespace simpla {