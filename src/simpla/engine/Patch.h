//
// Created by salmon on 16-12-8.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <simpla/SIMPLA_config.h>
#include <memory>

#include "SPObject.h"

namespace simpla {
namespace engine {
class MeshBlock;
class DataBlock;
class Patch {
    SP_OBJECT_BASE(Patch)
   public:
    Patch();
    virtual ~Patch();
    id_type GetMeshBlockId() const;
    std::shared_ptr<MeshBlock> GetMeshBlock() const;
    std::shared_ptr<DataBlock> GetDataBlock(id_type const &id) const;
    std::map<id_type, std::shared_ptr<DataBlock>> &GetAllDataBlock() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_PATCH_H
