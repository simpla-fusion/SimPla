//
// Created by salmon on 17-2-13.
//

#ifndef SIMPLA_MESHBLOCK_H
#define SIMPLA_MESHBLOCK_H

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/sp_def.h>

namespace simpla {
namespace engine {
class MeshBlock {
   public:
    MeshBlock(index_box_type const &b, size_type level = 0);
    MeshBlock(MeshBlock const &);
    ~MeshBlock();
    id_type GetGUID() const;
    size_type GetLevel() const;
    size_tuple GetDimensions() const;
    index_tuple GetOffset() const;
    index_tuple GetGhostWidth() const;
    index_box_type GetIndexBox(int IFORM = 0, int sub = 0) const;
    index_box_type GetOuterIndexBox(int IFORM = 0, int sub = 0) const;
    index_box_type GetInnerIndexBox(int IFORM = 0, int sub = 0) const;

    box_type GetBoundBox() const;
    size_type size(int IFORM = VERTEX) const { return 0; }
    size_tuple dimensions() const { return size_tuple{}; };
   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}
#endif  // SIMPLA_MESHBLOCK_H
