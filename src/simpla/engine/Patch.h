//
// Created by salmon on 16-12-8.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include "simpla/SIMPLA_config.h"

#include <memory>

#include "simpla/algebra/EntityId.h"
#include "simpla/data/Data.h"

#include "MeshBlock.h"
#include "SPObject.h"
namespace simpla {
namespace engine {

class Patch {
    SP_OBJECT_BASE(Patch)
   public:
    explicit Patch(MeshBlock const &blk);
    explicit Patch(MeshBlock &&blk);
    virtual ~Patch();

    Patch(this_type const &other);
    Patch(this_type &&other) noexcept;
    Patch &operator=(this_type const &other);
    this_type &operator=(this_type &&other) noexcept;

    void swap(Patch &other);

    void SetMeshBlock(const MeshBlock &);
    const MeshBlock * GetMeshBlock() const;

    void SetDataBlock(id_type id, std::shared_ptr<data::DataBlock> const &);
    std::shared_ptr<data::DataBlock> GetDataBlock(id_type const &id);

    struct DataPack_s {
        DataPack_s() = default;
        virtual ~DataPack_s() = default;
    };
    void SetDataPack(std::shared_ptr<DataPack_s> const &p) { m_pack_ = p; }
    std::shared_ptr<DataPack_s> GetDataPack() { return m_pack_; }

   private:
    MeshBlock m_block_;
    std::map<id_type, std::shared_ptr<data::DataBlock>> m_data_;
    std::shared_ptr<DataPack_s> m_pack_ = nullptr;
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_PATCH_H
