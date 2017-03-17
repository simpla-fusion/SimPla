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
    MeshBlock(index_box_type const &b, int level = 0);
    MeshBlock(MeshBlock const &);
    ~MeshBlock();
    id_type GetGUID() const;
    size_type GetLevel() const;
    size_tuple GetDimensions() const;
    index_tuple GetOffset() const;
    index_box_type const &GetIndexBox() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}
#endif  // SIMPLA_MESHBLOCK_H
