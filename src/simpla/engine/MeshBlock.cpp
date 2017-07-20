//
// Created by salmon on 17-3-1.
//
#include "simpla/SIMPLA_config.h"

#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include "simpla/algebra/EntityId.h"

#include "MeshBlock.h"
#include "SPObject.h"

namespace simpla {
namespace engine {

MeshBlock::MeshBlock(index_box_type b, size_type level, int owner)
    : m_owner_(owner), m_level_(level), m_index_box_(std::move(b)) {}

MeshBlock::~MeshBlock() = default;
MeshBlock::MeshBlock(MeshBlock const &other) = default;
//    : m_owner_(other.m_owner_), m_level_(other.m_level_), m_id_(other.m_id_), m_index_box_(other.m_index_box_) {}

MeshBlock::MeshBlock(MeshBlock &&other) noexcept
    : m_owner_(other.m_owner_),
      m_level_(other.m_level_),
      m_id_(other.m_id_),
      m_index_box_(std::move(other.m_index_box_)) {}

void MeshBlock::swap(MeshBlock &other) {
    std::swap(m_owner_, other.m_owner_);
    std::swap(m_level_, other.m_level_);
    std::swap(m_id_, other.m_id_);
    m_index_box_.swap(other.m_index_box_);
}

// MeshBlock &MeshBlock::operator=(MeshBlock const &other) {
//    MeshBlock(other).swap(*this);
//    return *this;
//}
// MeshBlock &MeshBlock::operator=(MeshBlock &&other) noexcept {
//    MeshBlock(other).swap(*this);
//    return *this;
//}
// index_tuple MeshBlock::GetGhostWidth() const { return m_ghost_width_; };
// index_box_type MeshBlock::GetOuterIndexBox() const {
//    auto ibox = GetIndexBox();
//    std::get<0>(ibox) -= GetGhostWidth();
//    std::get<1>(ibox) += GetGhostWidth();
//    return std::move(ibox);
//}
// index_box_type MeshBlock::GetInnerIndexBox() const { return GetIndexBox(); }
//
// index_tuple MeshBlock::GetIndexOrigin() const { return std::get<0>(m_index_box_); }
//
// size_tuple MeshBlock::GetDimensions() const {
//    return std::get<1>(m_index_box_) - std::get<0>(m_index_box_);
//}

}  // namespace engine {
}  // namespace simpla {