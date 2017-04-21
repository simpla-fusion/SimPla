//
// Created by salmon on 17-2-13.
//

#ifndef SIMPLA_MESHBLOCK_H
#define SIMPLA_MESHBLOCK_H

#include <simpla/algebra/nTupleExt.h>
#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/Range.h>
#include <simpla/utilities/sp_def.h>
namespace simpla {
namespace engine {

class MeshBlock {
   public:
    MeshBlock(index_box_type const &b, size_type level = 0, Real time_now = 0);
    MeshBlock(MeshBlock const &);
    ~MeshBlock();

    Real GetTime() const;
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
