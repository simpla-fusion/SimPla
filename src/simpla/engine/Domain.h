//
// Created by salmon on 17-2-10.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H
namespace simpla {
namespace engine {
class AttributeView;
class Patch;

class Domain {
   public:
    Domain() {}
    ~Domain(){};
    void Accept(Patch *);
    void Connect(AttributeView *attr) { m_attrs_.insert(attr); };
    void Disconnect(AttributeView *attr) { m_attrs_.erase(attr); }
    std::set<AttributeView *> m_observers_;
    std::unique_ptr<MeshView> m_mesh_;
};
}  // namespace engine {
}  // namespace simpla {
#endif  // SIMPLA_DOMAIN_H
