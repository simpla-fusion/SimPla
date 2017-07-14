//
// Created by salmon on 17-2-22.
//

#include "Patch.h"
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <map>
#include <memory>
#include "MeshBlock.h"
#include "simpla/SIMPLA_config.h"
#include "simpla/data/Data.h"

namespace simpla {
namespace engine {
struct Patch::pimpl_s {
    MeshBlock m_block_;
    std::map<id_type, std::shared_ptr<data::DataBlock>> m_data_;
    std::map<std::string, std::shared_ptr<PatchDataPack>> m_packs_;
};

Patch::Patch(id_type id) : m_pimpl_(new pimpl_s) {}
Patch::~Patch() {}
Patch::Patch(this_type const &other) : Patch(other.GetId()) {
    MeshBlock(other.GetMeshBlock()).swap(m_pimpl_->m_block_);
    m_pimpl_->m_data_ = other.m_pimpl_->m_data_;
    m_pimpl_->m_packs_ = other.m_pimpl_->m_packs_;
}
Patch::Patch(this_type &&other) noexcept : m_pimpl_(other.m_pimpl_.get()) { other.m_pimpl_.reset(); }

void Patch::swap(Patch &other) { std::swap(m_pimpl_, other.m_pimpl_); }

Patch &Patch::operator=(Patch const &other) {
    Patch(other).swap(*this);
    return *this;
}
Patch &Patch::operator=(Patch &&other) noexcept {
    Patch(other).swap(*this);
    return *this;
}

id_type Patch::GetId() const { return m_pimpl_->m_block_.GetGUID(); }

void Patch::SetMeshBlock(const MeshBlock &m) { MeshBlock(m).swap(m_pimpl_->m_block_); }

const MeshBlock &Patch::GetMeshBlock() const { return m_pimpl_->m_block_; }

//std::map<id_type, std::shared_ptr<data::DataBlock>> &Patch::GetAllData() { return m_pimpl_->m_data_; };

void Patch::SetDataBlock(id_type id, std::shared_ptr<data::DataBlock> d) { m_pimpl_->m_data_[id] = d; }
std::shared_ptr<data::DataBlock> Patch::GetDataBlock(id_type const &id) {
    std::shared_ptr<data::DataBlock> res = nullptr;
    auto it = m_pimpl_->m_data_.find(id);
    if (it != m_pimpl_->m_data_.end()) { res = (it->second); }
    return (res);
}
std::shared_ptr<PatchDataPack> Patch::GetPack(const std::string &g) {
    auto it = m_pimpl_->m_packs_.find(g);
    if (it != m_pimpl_->m_packs_.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

void Patch::SetPack(const std::string &g, std::shared_ptr<PatchDataPack> p) { m_pimpl_->m_packs_[g] = p; }

//
// void Patch::PushRange(std::shared_ptr<std::map<std::string, EntityRange>> const &r) { m_pimpl_->m_range_ = r; };
// std::shared_ptr<std::map<std::string, EntityRange>> Patch::PopRange() { return m_pimpl_->m_range_; };

}  // namespace engine {
}  // namespace simpla {