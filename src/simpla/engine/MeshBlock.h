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

   protected:
    explicit MeshBlock(index_box_type b, int id, int level, int global_rank);

   public:
    ~MeshBlock();
    SP_DEFAULT_CONSTRUCT(MeshBlock);
    static std::shared_ptr<MeshBlock> New(index_box_type const &box, int id = 0, int level = 0, int global_rank = 0);

   private:
    static constexpr int MAX_LEVEL_NUMBER = 8;
    static constexpr int MAX_LOCAL_ID_NUMBER = std::numeric_limits<int>::max() / 2;

   public:
    bool operator==(MeshBlock const &other) const {
        return m_level_ == other.m_level_ && m_index_box_ == other.m_index_box_;
    };

    int GetOwnerRank() const { return m_global_rank_; }
    int GetLevel() const { return m_level_; }
    int GetLocalID() const { return m_local_id_; }

    id_type GetGUID() const;

    index_box_type IndexBox() const { return m_index_box_; }

    static id_type hash_id(int id = 0, int level = 0, int owner = 0);

   private:
    int m_global_rank_ = 0;
    int m_level_ = 0;
    int m_local_id_ = 0;
    index_box_type m_index_box_{{0, 0, 0}, {1, 1, 1}};
};  // class MeshBlock

}  // namespace engine{
}  // namespace simpla

#endif  // SIMPLA_MESHBLOCK_H
