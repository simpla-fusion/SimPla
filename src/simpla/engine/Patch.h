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
template <typename>
class Range;
namespace geometry {
class GeoObject;
}
namespace engine {
class MeshBlock;
class Patch {
    SP_OBJECT_BASE(Patch)
   public:
    Patch();
    //    Patch(Patch const &);
    //    Patch(Patch &&);
    virtual ~Patch();

    id_type GetBlockId() const;
    void SetBlock(std::shared_ptr<MeshBlock> const &);
    std::shared_ptr<MeshBlock> GetBlock() const;

    int Push(id_type const &id, std::shared_ptr<data::DataBlock> const &);
    std::shared_ptr<data::DataBlock> Pop(id_type const &id) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_PATCH_H
