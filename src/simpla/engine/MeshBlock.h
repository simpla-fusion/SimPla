//
// Created by salmon on 17-2-13.
//

#ifndef SIMPLA_MESHBLOCK_H
#define SIMPLA_MESHBLOCK_H

#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/Range.h>
#include <simpla/utilities/sp_def.h>
namespace simpla {
namespace engine {

class MeshBlock {
    SP_OBJECT_BASE(MeshBlock)
   public:
    explicit MeshBlock(index_box_type const &b, size_type level = 0, point_type const &x0 = {0, 0, 0},
                       point_type const &dx = {1, 1, 1});
    ~MeshBlock();

    SP_DEFAULT_CONSTRUCT(MeshBlock)
    id_type GetGUID() const;
    size_type GetLevel() const;
    size_tuple GetDimensions() const;
    index_tuple GetOffset() const;
    index_tuple GetGhostWidth() const;
    index_box_type GetIndexBox() const;
    index_box_type GetOuterIndexBox() const;
    index_box_type GetInnerIndexBox() const;


   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}
#endif  // SIMPLA_MESHBLOCK_H
