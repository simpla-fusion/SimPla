//
// Created by salmon on 16-12-8.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include "simpla/SIMPLA_config.h"

#include <memory>

#include "simpla/algebra/EntityId.h"
#include "simpla/data/Data.h"

#include "SPObject.h"

namespace simpla {
namespace engine {
class MeshBlock;

class Patch {
    SP_OBJECT_BASE(Patch)
   public:
    explicit Patch(id_type id = NULL_ID);
    virtual ~Patch();
    Patch(this_type const &other);
    Patch(this_type &&other) noexcept;
    Patch &operator=(this_type const &other);
    this_type &operator=(this_type &&other) noexcept;
    void swap(Patch &other);

    bool empty() const;

    id_type GetId() const;

    void SetBlock(const MeshBlock &);
    const MeshBlock &GetBlock() const;

    std::map<id_type, std::shared_ptr<data::DataBlock>> &GetAllData();

    void Push(id_type id, std::shared_ptr<data::DataBlock>);
    std::shared_ptr<data::DataBlock> Pop(id_type const &id);

    EntityRange GetRange(const std::string &g) const;
    EntityRange &GetRange(const std::string &g);
    EntityRange &AddRange(const std::string &g, EntityRange &&r);

    std::shared_ptr<std::map<std::string, EntityRange>> GetRanges();
    void SetRanges(std::shared_ptr<std::map<std::string, EntityRange>> const &);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_PATCH_H
