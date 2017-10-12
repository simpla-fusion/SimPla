//
// Created by salmon on 17-10-12.
//

#include "Patch.h"
namespace simpla {
namespace engine {
struct Patch::pimpl_s {
    std::shared_ptr<const MeshBlock> m_mesh_block_;
    std::map<std::string, std::shared_ptr<data::DataNode>> m_data_blocks_;
};

Patch::Patch() : m_pimpl_(new pimpl_s) {}
Patch::~Patch() { delete m_pimpl_; }
Patch::Patch(std::shared_ptr<const MeshBlock> const &mblk) : Patch() { m_pimpl_->m_mesh_block_ = mblk; }

id_type Patch::GetGUID() const { return m_pimpl_->m_mesh_block_->GetGUID(); }

std::shared_ptr<data::DataNode> Patch::Serialize() const {
    auto res = data::DataNode::New(data::DataNode::DN_TABLE);
    FIXME;
    return res;
}
void Patch::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { FIXME; }

std::map<std::string, std::shared_ptr<data::DataNode>> const &Patch::GetAllDataBlocks() const {
    return m_pimpl_->m_data_blocks_;
}

std::shared_ptr<data::DataNode> Patch::GetDataBlock(std::string const &key) const {
    auto it = m_pimpl_->m_data_blocks_.find(key);
    return it == m_pimpl_->m_data_blocks_.end() ? nullptr : it->second;
}
void Patch::SetDataBlock(std::string const &key, std::shared_ptr<data::DataNode> const &data) {
    auto res = m_pimpl_->m_data_blocks_.emplace(key, data);
    if (!res.second) { res.first->second->Set(data); }
}

void Patch::SetMeshBlock(const std::shared_ptr<const MeshBlock> &blk) { m_pimpl_->m_mesh_block_ = blk; }
std::shared_ptr<const MeshBlock> Patch::GetMeshBlock() const { return m_pimpl_->m_mesh_block_; }

void Patch::Push(std::shared_ptr<Patch> const &other) {
    if (other == nullptr) { return; }
    ASSERT(GetGUID() == other->GetGUID());
    for (auto const &item : other->m_pimpl_->m_data_blocks_) { SetDataBlock(item.first, item.second); }
}
std::shared_ptr<Patch> Patch::Pop() const { return const_cast<Patch *>(this)->shared_from_this(); }
}  // namespace engine
}  // namespace simpla