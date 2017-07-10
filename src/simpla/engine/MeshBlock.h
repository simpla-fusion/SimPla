//
// Created by salmon on 17-2-13.
//

#ifndef SIMPLA_MESHBLOCK_H
#define SIMPLA_MESHBLOCK_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTuple.ext.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/utilities/ObjectHead.h>
#include <memory>

namespace simpla {
namespace engine {

class MeshBlock {
    SP_OBJECT_BASE(MeshBlock)
   public:
    MeshBlock();
    explicit MeshBlock(index_box_type const &b, size_type level);
    ~MeshBlock();

    MeshBlock(MeshBlock const &other);
    MeshBlock(MeshBlock &&other) noexcept;
    void swap(MeshBlock &);

    MeshBlock &operator=(MeshBlock const &other);
    MeshBlock &operator=(MeshBlock &&other) noexcept;

    int GetLevel() const;
    id_type GetGUID() const;

    size_tuple GetDimensions() const;
    index_tuple GetIndexOrigin() const;
    index_tuple GetGhostWidth() const;
    index_box_type GetIndexBox() const;
    index_box_type GetOuterIndexBox() const;
    index_box_type GetInnerIndexBox() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};  // class MeshBlock

}  // namespace engine{
}  // namespace simpla

#endif  // SIMPLA_MESHBLOCK_H
