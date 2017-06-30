//
// Created by salmon on 16-12-8.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <simpla/SIMPLA_config.h>
#include <simpla/data/all.h>
#include <simpla/geometry/GeoObject.h>
#include <simpla/utilities/EntityId.h>
#include <memory>
#include "simpla/utilities/SPObject.h"
#include "../../../cmake-build-debug/include/simpla/SIMPLA_config.h"

namespace simpla {

namespace engine {
class MeshBlock;
class Patch {
    SP_OBJECT_BASE(Patch)
   public:
    explicit Patch(id_type id = NULL_ID);
    virtual ~Patch();

    SP_DEFAULT_CONSTRUCT(Patch);
    id_type GetId() const;

    void SetBlock(std::shared_ptr<MeshBlock> const &);
    std::shared_ptr<MeshBlock> GetBlock() const;
    void Merge(Patch &other);
    std::map<id_type, std::shared_ptr<data::DataBlock>> &GetAllData();

    void Push(id_type id, std::shared_ptr<data::DataBlock> const &);
    std::shared_ptr<data::DataBlock> Pop(id_type const &id) const;

    std::map<std::string, EntityRange> m_ranges_;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_PATCH_H
