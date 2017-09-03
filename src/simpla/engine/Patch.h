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
namespace simpla {
namespace engine {

class Patch {
    SP_OBJECT_BASE(Patch)
   protected:
    explicit Patch(std::shared_ptr<MeshBlock> const &blk);

   public:
    virtual ~Patch();
    SP_DEFAULT_CONSTRUCT(Patch);
    static std::shared_ptr<Patch> New(std::shared_ptr<MeshBlock> const &);

    void SetMeshBlock(const std::shared_ptr<MeshBlock> &);
    std::shared_ptr<MeshBlock> GetMeshBlock() const;

    void SetDataBlock(id_type id, std::shared_ptr<data::DataBlock> const &);
    std::shared_ptr<data::DataBlock> GetDataBlock(id_type const &id);

    struct DataPack_s {
        DataPack_s(){};
        virtual ~DataPack_s(){};
    };
    void SetDataPack(std::shared_ptr<DataPack_s> const &p) { m_pack_ = p; }
    std::shared_ptr<DataPack_s> GetDataPack() { return m_pack_; }

   private:
    std::shared_ptr<MeshBlock> m_block_;
    std::map<id_type, std::shared_ptr<data::DataBlock>> m_data_;
    std::shared_ptr<DataPack_s> m_pack_ = nullptr;
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_PATCH_H
