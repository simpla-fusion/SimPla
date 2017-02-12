//
// Created by salmon on 17-2-12.
//

#ifndef SIMPLA_DOMAINVIEW_H
#define SIMPLA_DOMAINVIEW_H

#include <simpla/SIMPLA_config.h>
#include <memory>
#include <set>
namespace simpla {
namespace mesh {
class MeshView;
}
namespace engine {
class Domain;
class AttributeView;
class DataBlock;

class DomainView {
   public:
    void Dispatch(Domain const &d);

    void Connect(AttributeView *attr);
    void Disconnect(AttributeView *attr);
    std::shared_ptr<DataBlock> data_block(id_type) const;
    void data_block(id_type, std::shared_ptr<DataBlock>);

    virtual mesh::MeshView *mesh() { return m_mesh_; }

   private:
    mesh::MeshView *m_mesh_;
    std::set<AttributeView *> m_attrs_;
};
}
}
#endif  // SIMPLA_DOMAINVIEW_H
