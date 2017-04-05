//
// Created by salmon on 16-12-8.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <simpla/SIMPLA_config.h>
#include <simpla/data/all.h>
#include <memory>
#include "SPObject.h"

namespace simpla {
namespace engine {
class MeshBlock;
class Patch {
    SP_OBJECT_BASE(Patch)
   public:
    Patch();
    virtual ~Patch();
    id_type GetBlockId() const;
    std::shared_ptr<MeshBlock> const & GetMeshBlock()const;
    void PushMeshBlock(std::shared_ptr<MeshBlock> const &);
    std::shared_ptr<MeshBlock> PopMeshBlock();
    int PushData(id_type const &id, std::shared_ptr<data::DataBlock> const &);
    std::shared_ptr<data::DataBlock> PopData(id_type const &id);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_PATCH_H
