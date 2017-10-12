//
// Created by salmon on 17-10-12.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <memory>

#include <simpla/data/DataNode.h>
#include "MeshBlock.h"

namespace simpla {
namespace engine {
class Patch : public std::enable_shared_from_this<Patch> {
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;

   private:
    Patch();
    explicit Patch(std::shared_ptr<const MeshBlock> const &);

   public:
    ~Patch();
    Patch(Patch const &) = delete;

    id_type GetGUID() const;

    template <typename... Args>
    static std::shared_ptr<Patch> New(Args &&... args) {
        return std::shared_ptr<Patch>(new Patch(std::forward<Args>(args)...));
    };
    std::shared_ptr<data::DataNode> Serialize() const;
    void Deserialize(std::shared_ptr<data::DataNode> const &cfg);

    std::shared_ptr<data::DataNode> GetDataBlock(std::string const &) const;
    void SetDataBlock(std::string const &, std::shared_ptr<data::DataNode> const &);
    std::map<std::string, std::shared_ptr<data::DataNode>> const &GetAllDataBlocks() const;

    void SetMeshBlock(const std::shared_ptr<const MeshBlock> &blk);
    std::shared_ptr<const MeshBlock> GetMeshBlock() const;
    index_box_type GetIndexBox() const;

    void Push(std::shared_ptr<Patch> const &other);
    std::shared_ptr<Patch> Pop() const;
};
}  // namespace engine
}
#endif  // SIMPLA_PATCH_H
