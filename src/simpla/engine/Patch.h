//
// Created by salmon on 16-12-8.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <simpla/SIMPLA_config.h>
#include <map>
#include "DataBlock.h"
#include "MeshBlock.h"
namespace simpla {
namespace engine {

class Patch {
    SP_OBJECT_BASE(Patch)
   public:
    Patch() {}
    virtual ~Patch() {}

    std::shared_ptr<MeshBlock> const &mesh_block() const { return m_mesh_; }
    void mesh_block(std::shared_ptr<MeshBlock> const &m) { m_mesh_ = m; }

    virtual std::shared_ptr<DataBlock> data(id_type const &id, std::shared_ptr<DataBlock> const &p = (nullptr)) {
        return m_data_.emplace(id, p).first->second;
    }
    virtual std::shared_ptr<DataBlock> data(id_type const &id) const {
        auto it = m_data_.find(id);
        return (it != m_data_.end()) ? it->second : std::shared_ptr<DataBlock>(nullptr);
    }

   private:
    std::shared_ptr<MeshBlock> m_mesh_;
    std::map<id_type, std::shared_ptr<DataBlock> > m_data_;
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_PATCH_H
