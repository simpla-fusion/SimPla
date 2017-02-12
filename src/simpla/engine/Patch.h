//
// Created by salmon on 16-12-8.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <simpla/SIMPLA_config.h>
#include <map>
#include "DataBlock.h"

namespace simpla {
namespace mesh {

class Patch : public Object {
    SP_OBJECT_HEAD(Patch, Object)

   public:
    std::shared_ptr<MeshBlock> const &mesh_block() const { return m_mesh_; }

    void mesh_block(std::shared_ptr<MeshBlock> const &m) { m_mesh_ = m; }

    std::shared_ptr<DataBlock> data(id_type const &id, std::shared_ptr<DataBlock> const &p = (nullptr)) {
        return m_data_.emplace(id, p).first->second;
    }

    std::shared_ptr<DataBlock> data(id_type const &id) const {
        auto it = m_data_.find(id);
        if (it != m_data_.end()) {
            return it->second;
        } else {
            return std::shared_ptr<DataBlock>(nullptr);
        }
    }

    template <typename U>
    U const *data_as(id_type const &n) const {
        auto d = data(n);
        ASSERT(d->isA(typeid(U)));
        return static_cast<U *>(d.get());
    }

    template <typename U>
    U *data_as(id_type const &n, std::shared_ptr<U> const &p = nullptr) {
        auto &d = data(n, p);
        ASSERT(d->isA(typeid(U)));
        return static_cast<U *>(d.get());
    }

   private:
    std::shared_ptr<MeshBlock> m_mesh_;
    std::map<id_type, std::shared_ptr<DataBlock> > m_data_;
};
}
}  // namespace simpla { namespace mesh

#endif  // SIMPLA_PATCH_H
