//
// Created by salmon on 17-3-1.
//

#include "MeshBlock.h"
#include "SPObject.h"

namespace simpla {
namespace engine {

struct MeshBlock::pimpl_s {
    int m_level_ = 0;
    id_type m_GUID_ = NULL_ID;
    size_tuple m_dimensions_{1, 1, 1};
    index_tuple m_offset_{0, 0, 0};
    box_type m_box_;
};
MeshBlock::MeshBlock() : m_pimpl_(new pimpl_s) {}
MeshBlock::~MeshBlock() {}
size_tuple const &MeshBlock::GetDimensions() const { return m_pimpl_->m_dimensions_; }
index_tuple const &MeshBlock::GetOffset() const { return m_pimpl_->m_offset_; }
box_type const &MeshBlock::GetBox() const { return m_pimpl_->m_box_; }
id_type MeshBlock::GetGUID() const { return m_pimpl_->m_GUID_; }
int MeshBlock::GetLevel() const { return m_pimpl_->m_level_; }

}  // namespace engine {
}  // namespace simpla {