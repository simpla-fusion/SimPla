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
struct Domain : public AttributeHolder
{

public:

    Domain() {}


    template<typename ...Args>
    explicit Domain(Args &&...args):m_mesh_block_(std::make_shared<MeshBlock>(std::forward<Args>(args)...)) {}

    virtual ~Domain() {}

    virtual void initialize() { DO_NOTHING; }

    virtual void deploy() { DO_NOTHING; }

    void move_to(std::shared_ptr<MeshBlock> const &m) { m_mesh_block_ = m; }

    std::shared_ptr<MeshBlock> &mesh_block() { return m_mesh_block_; }

    std::shared_ptr<MeshBlock> const &mesh_block() const { return m_mesh_block_; }

private:
    std::shared_ptr<MeshBlock> m_mesh_block_;

};
}}
#endif //SIMPLA_GEOMETRY_H
