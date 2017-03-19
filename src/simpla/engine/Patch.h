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
    Patch(std::shared_ptr<MeshBlock> const &p = nullptr);
    virtual ~Patch();
    id_type GetMeshBlockId() const;
    void SetMeshBlock(std::shared_ptr<MeshBlock> const &);
    std::shared_ptr<MeshBlock> GetMeshBlock() const;
    int SetDataBlock(id_type const &id, std::shared_ptr<data::DataEntity> const &);
    std::shared_ptr<data::DataEntity> GetDataBlock(id_type const &id) const;
    std::map<id_type, std::shared_ptr<data::DataEntity>> &GetAllDataBlock() const;
    void Push(std::shared_ptr<Patch> const &) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_PATCH_H
