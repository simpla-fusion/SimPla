//
// Created by salmon on 17-10-12.
//

#include "Patch.h"
namespace simpla {
namespace engine {
struct Patch::pimpl_s {
    std::shared_ptr<const MeshBlock> m_mesh_block_;
    std::map<std::string, std::shared_ptr<data::DataEntry>> m_data_blocks_;
};

Patch::Patch() : m_pimpl_(new pimpl_s) {}
Patch::~Patch() { delete m_pimpl_; }
Patch::Patch(std::shared_ptr<const MeshBlock> const &mblk) : Patch() { m_pimpl_->m_mesh_block_ = mblk; }

id_type Patch::GetGUID() const { return m_pimpl_->m_mesh_block_->GetGUID(); }

std::shared_ptr<data::DataEntry> Patch::Serialize() const {
    auto res = data::DataEntry::Create(data::DataEntry::DN_TABLE);
    res->Set("MeshBlock", m_pimpl_->m_mesh_block_->Serialize());

    auto attrs = res->CreateNode("Attributes", data::DataEntry::DN_TABLE);
    for (auto const &item : m_pimpl_->m_data_blocks_) { attrs->Set(item.first, item.second); }
    return res;
}
void Patch::Deserialize(std::shared_ptr<const data::DataEntry> const &cfg) {
    m_pimpl_->m_mesh_block_ = MeshBlock::New(cfg->Get("MeshBlock"));
    if (auto attrs = cfg->Get("Attributes")) {
        attrs->Foreach([&](std::string const &key, std::shared_ptr<const data::DataEntry> const &node) {
            auto res = m_pimpl_->m_data_blocks_.emplace(key, node->Copy());
            if (!res.second) { res.first->second->Set(node); }
        });
    }
}

std::map<std::string, std::shared_ptr<data::DataEntry>> const &Patch::GetAllDataBlocks() const {
    return m_pimpl_->m_data_blocks_;
}

std::shared_ptr<data::DataEntry> Patch::GetDataBlock(std::string const &key) const {
    std::shared_ptr<data::DataEntry> res = nullptr;
    auto it = m_pimpl_->m_data_blocks_.find(key);
    if (it != m_pimpl_->m_data_blocks_.end()) { res = it->second; }
    return res;
}

void Patch::SetDataBlock(std::string const &key, std::shared_ptr<data::DataEntry> const &d) {
    auto res = m_pimpl_->m_data_blocks_.emplace(key, d);
    res.first->second->Set(d);
}

void Patch::SetMeshBlock(const std::shared_ptr<const MeshBlock> &blk) { m_pimpl_->m_mesh_block_ = blk; }
std::shared_ptr<const MeshBlock> Patch::GetMeshBlock() const { return m_pimpl_->m_mesh_block_; }
index_box_type Patch::GetIndexBox() const { return m_pimpl_->m_mesh_block_->GetIndexBox(); }

void Patch::Push(std::shared_ptr<Patch> const &other) {
    if (other == nullptr) { return; }
    ASSERT(GetGUID() == other->GetGUID());
    for (auto const &item : other->m_pimpl_->m_data_blocks_) { SetDataBlock(item.first, item.second); }
}
std::shared_ptr<Patch> Patch::Pop() const { return const_cast<Patch *>(this)->shared_from_this(); }
}  // namespace engine
}  // namespace simpla