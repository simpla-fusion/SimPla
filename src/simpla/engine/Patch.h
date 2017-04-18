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
class Chart;
class Patch {
    SP_OBJECT_BASE(Patch)
   public:
    Patch();
    Patch(Patch const &) = delete;
    Patch(Patch &&) = delete;
    virtual ~Patch();

    void SetChart(std::shared_ptr<Chart>);
    std::shared_ptr<Chart> GetChart() const;

    id_type GetBlockId() const;
    void SetBlock(std::shared_ptr<MeshBlock> const &);
    std::shared_ptr<MeshBlock> GetBlock() const;

    std::map<id_type, std::shared_ptr<data::DataBlock>> &GetAllData();

    int Push(id_type const &id, std::shared_ptr<data::DataBlock> const &);
    std::shared_ptr<data::DataBlock> Pop(id_type const &id) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_PATCH_H
