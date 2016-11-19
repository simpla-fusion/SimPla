//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_GEOMETRY_H
#define SIMPLA_GEOMETRY_H

#include <simpla/toolbox/design_pattern/Observer.h>
#include <simpla/mesh/Attribute.h>
#include "simpla/mesh/MeshBlock.h"
#include "simpla/mesh/DataBlock.h"

namespace simpla { namespace mesh
{
struct GeometryBase : public design_pattern::Observer<void(std::shared_ptr<MeshBlock> const &)>
{
private:
    typedef design_pattern::Observer<void(std::shared_ptr<MeshBlock> const &)> base_type;
public:
    std::shared_ptr<MeshBlock> m_mesh_ = nullptr;

    GeometryBase() : m_mesh_(nullptr) {}

    template<typename ...Args>
    explicit GeometryBase(Args &&...args):m_mesh_(new MeshBlock(std::forward<Args>(args)...)) {}

    virtual ~GeometryBase() {}

    virtual void connect(mesh::AttributeHolder *subject) { base_type::connect(subject); }

    virtual void notify(std::shared_ptr<MeshBlock> const &m) { move_to(m); };

    virtual void move_to(std::shared_ptr<MeshBlock> const &m)
    {
        m_mesh_ = m;
        deploy();
    }

    virtual void initialize() { DO_NOTHING; }

    virtual void deploy() { DO_NOTHING; }

    std::shared_ptr<mesh::MeshBlock> mesh_block()
    {
        ASSERT(m_mesh_ != nullptr);
        return m_mesh_;
    }

    std::shared_ptr<mesh::MeshBlock> mesh_block() const
    {
        ASSERT(m_mesh_ != nullptr);
        return m_mesh_;
    }
};
}}
#endif //SIMPLA_GEOMETRY_H
