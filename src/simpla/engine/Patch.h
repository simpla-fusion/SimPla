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
namespace geometry {
class GeoObject;
}
namespace engine {
class MeshBlock;
class Patch {
    SP_OBJECT_BASE(Patch)
   public:
    Patch();
    virtual ~Patch();
    id_type GetBlockId() const;
    std::shared_ptr<MeshBlock> const &GetMeshBlock() const;
    void PushMeshBlock(std::shared_ptr<MeshBlock> const &);
    std::shared_ptr<MeshBlock> PopMeshBlock();
    int Push(id_type const &id, std::shared_ptr<data::DataBlock> const &);
    std::shared_ptr<data::DataBlock> Pop(id_type const &id);

    void SetGeoObject(std::shared_ptr<geometry::GeoObject> const &g);
    std::shared_ptr<geometry::GeoObject> const &GetGeoObject() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_PATCH_H
