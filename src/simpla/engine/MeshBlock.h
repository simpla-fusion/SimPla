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
    template <typename... Args>
    MeshBlock(Args &&... args) {}
    MeshBlock();
    ~MeshBlock();
    id_type GetGUID() const;
    int GetLevel() const;
    size_tuple const &GetDimensions() const;
    index_tuple const &GetOffset() const;
    box_type const &GetBox() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}
#endif  // SIMPLA_MESHBLOCK_H
