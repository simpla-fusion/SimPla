//
// Created by salmon on 17-2-13.
//

#ifndef SIMPLA_MESHBLOCK_H
#define SIMPLA_MESHBLOCK_H

#include "simpla/SIMPLA_config.h"

#include <memory>

#include "simpla/algebra/nTuple.ext.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/utilities/ObjectHead.h"

namespace simpla {
namespace engine {

class MeshBlock {
    SP_OBJECT_BASE(MeshBlock)
   public:
    explicit MeshBlock(index_box_type b = index_box_type{{0, 0, 0}, {1, 1, 1}}, size_type level = 0, int owner = 0);
    ~MeshBlock();

    MeshBlock(MeshBlock const &other);
    MeshBlock(MeshBlock &&other) noexcept;
    void swap(MeshBlock &);

    MeshBlock &operator=(MeshBlock const &other) {
        this_type(other).swap(*this);
        return *this;
    };
    MeshBlock &operator=(MeshBlock &&other) noexcept {
        this_type(std::forward<MeshBlock>(other)).swap(*this);
        return *this;
    }

    bool operator==(MeshBlock const &other) const {
        return m_level_ == other.m_level_ && m_index_box_ == other.m_index_box_;
    };

    int GetOwnerRank() const { return m_owner_; }
    size_type GetLevel() const { return m_level_; }
    index_box_type GetIndexBox() const { return m_index_box_; }

    id_type GetID() const { return m_id_; }

   private:
    int m_owner_ = 0;
    size_type m_level_ = 0;
    index_box_type m_index_box_{{0, 0, 0}, {1, 1, 1}};
    id_type m_id_;
};  // class MeshBlock

}  // namespace engine{
}  // namespace simpla

#endif  // SIMPLA_MESHBLOCK_H
