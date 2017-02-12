//
// Created by salmon on 17-2-10.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include <memory>
#include <set>
namespace simpla {
namespace mesh {
class MeshView;
class MeshBlock;
}
namespace engine {
class AttributeView;
class Patch;

class Domain {
   public:
    Domain() {}
    ~Domain(){};

    mesh::MeshBlock const &mesh_block() const {}
    std::set<AttributeView *> m_observers_;
    std::unique_ptr<MeshView> m_mesh_;
};
}  // namespace engine {
}  // namespace simpla {
#endif  // SIMPLA_DOMAIN_H
