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

Patch::Patch(std::shared_ptr<MeshBlock> const &blk) : m_block_(blk) {}
Patch::~Patch() = default;
std::shared_ptr<Patch> Patch::New(std::shared_ptr<MeshBlock> const &blk) {
    return std::shared_ptr<Patch>(new Patch(blk));
}

void Patch::SetMeshBlock(const std::shared_ptr<MeshBlock> &m) { m_block_ = m; }
std::shared_ptr<MeshBlock> Patch::GetMeshBlock() const { return m_block_; }

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