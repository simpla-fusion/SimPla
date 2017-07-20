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

Patch::Patch(MeshBlock const &blk) : m_block_(blk) {}
Patch::Patch(MeshBlock &&blk) : m_block_(std::move(blk)) {}

Patch::~Patch() = default;
Patch::Patch(this_type const &other) : m_block_(other.m_block_), m_data_(other.m_data_), m_pack_(m_pack_) {}
Patch::Patch(this_type &&other) noexcept
    : m_block_(std::move(other.m_block_)), m_data_(std::move(other.m_data_)), m_pack_(std::move(m_pack_)) {}

void Patch::swap(Patch &other) {
    std::swap(m_data_, other.m_data_);
    std::swap(m_pack_, m_pack_);
    std::swap(m_block_, other.m_block_);
}

Patch &Patch::operator=(Patch const &other) {
    Patch(other).swap(*this);
    return *this;
}
Patch &Patch::operator=(Patch &&other) noexcept {
    Patch(std::forward<Patch>(other)).swap(*this);
    return *this;
}

void Patch::SetMeshBlock(const MeshBlock &m) { MeshBlock(m).swap(m_block_); }

const MeshBlock &Patch::GetMeshBlock() const { return m_block_; }

void Patch::SetDataBlock(id_type id, std::shared_ptr<data::DataBlock> const &d) {
    auto res = m_data_.emplace(id, d);
    if (!res.second) { res.first->second = d; }
}
std::shared_ptr<data::DataBlock> Patch::GetDataBlock(id_type const &id) {
    std::shared_ptr<data::DataBlock> res = nullptr;
    auto it = m_data_.find(id);
    if (it != m_data_.end()) { res = (it->second); }
    return (res);
}

}  // namespace engine {
}  // namespace simpla {