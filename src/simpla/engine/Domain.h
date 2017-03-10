//
// Created by salmon on 17-2-10.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include <simpla/SIMPLA_config.h>
#include <memory>
#include <set>
#include <map>

namespace simpla {
namespace engine {
class AttributeView;
class MeshView;
class MeshBlock;
class DataBlock;
class Domain {
   public:
    Domain() {}
    ~Domain(){};

    MeshBlock const &mesh_block() const {}
    std::map<id_type, std::shared_ptr<DataBlock>> m_data_blocks_;
    std::shared_ptr<MeshBlock> m_mesh_;
};
}  // namespace engine {
}  // namespace simpla {
#endif  // SIMPLA_DOMAIN_H
