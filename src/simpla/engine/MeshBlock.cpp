//
// Created by salmon on 17-3-1.
//

#include "MeshBlock.h"
#include <simpla/algebra/EntityId.h>
#include <simpla/utilities/SPObject.h>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
namespace simpla {
namespace engine {

struct MeshBlock::pimpl_s {
    size_type m_level_ = 0;
    Real m_time_ = 0;
    id_type m_GUID_ = NULL_ID;
    index_box_type m_index_box_;
    index_tuple m_ghost_width_{4, 4, 4};
    static boost::uuids::random_generator m_gen_;
    static boost::hash<boost::uuids::uuid> m_hasher_;
};
boost::uuids::random_generator MeshBlock::pimpl_s::m_gen_;
boost::hash<boost::uuids::uuid> MeshBlock::pimpl_s::m_hasher_;

MeshBlock::MeshBlock() : m_pimpl_(new pimpl_s) {}
MeshBlock::MeshBlock(index_box_type const &b, size_type level) : MeshBlock() {
    m_pimpl_->m_index_box_ = b;
    m_pimpl_->m_level_ = level;
    m_pimpl_->m_GUID_ = m_pimpl_->m_hasher_(m_pimpl_->m_gen_());
};

MeshBlock::~MeshBlock() {}
MeshBlock::MeshBlock(MeshBlock const &other) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_level_ = other.m_pimpl_->m_level_;
    m_pimpl_->m_GUID_ = other.m_pimpl_->m_GUID_;
    m_pimpl_->m_index_box_ = other.m_pimpl_->m_index_box_;
    m_pimpl_->m_time_ = other.m_pimpl_->m_time_;
}
MeshBlock::MeshBlock(MeshBlock &&other) noexcept : m_pimpl_(other.m_pimpl_.get()) { other.m_pimpl_.reset(); }
void MeshBlock::swap(MeshBlock &other) { std::swap(other.m_pimpl_, m_pimpl_); }

MeshBlock &MeshBlock::operator=(MeshBlock const &other) {
    MeshBlock(other).swap(*this);
    return *this;
}
MeshBlock &MeshBlock::operator=(MeshBlock &&other) noexcept {
    MeshBlock(other).swap(*this);
    return *this;
}

index_tuple MeshBlock::GetGhostWidth() const { return m_pimpl_->m_ghost_width_; };

index_box_type MeshBlock::GetIndexBox() const { return m_pimpl_->m_index_box_; }

index_box_type MeshBlock::GetOuterIndexBox() const {
    auto ibox = GetIndexBox();
    std::get<0>(ibox) -= GetGhostWidth();
    std::get<1>(ibox) += GetGhostWidth();
    return std::move(ibox);
}
index_box_type MeshBlock::GetInnerIndexBox() const { return GetIndexBox(); }

index_tuple MeshBlock::GetIndexOrigin() const { return std::get<0>(m_pimpl_->m_index_box_); }

size_tuple MeshBlock::GetDimensions() const {
    return std::get<1>(m_pimpl_->m_index_box_) - std::get<0>(m_pimpl_->m_index_box_);
}

id_type MeshBlock::GetGUID() const { return m_pimpl_->m_GUID_; }
int MeshBlock::GetLevel() const { return m_pimpl_->m_level_; }

}  // namespace engine {
}  // namespace simpla {