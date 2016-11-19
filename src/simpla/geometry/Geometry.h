//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_GEOMETRY_H
#define SIMPLA_GEOMETRY_H

#include <simpla/toolbox/design_pattern/Observer.h>
#include "simpla/mesh/MeshBlock.h"
#include "simpla/mesh/DataBlock.h"

namespace simpla { namespace mesh
{
struct GeometryBase : public design_pattern::Observer<void(std::shared_ptr<MeshBlock> const &)>
{
    std::shared_ptr<MeshBlock> m_mesh_;

    template<typename ...Args>
    explicit GeometryBase(Args &&...args):m_mesh_(new MeshBlock(std::forward<Args>(args)...)) {}

    virtual void notify(std::shared_ptr<MeshBlock> const &m) { UNIMPLEMENTED; };

    std::shared_ptr<mesh::MeshBlock> mesh_block() { return m_mesh_; }

    std::shared_ptr<mesh::MeshBlock> mesh_block() const { return m_mesh_; }

};
}}
#endif //SIMPLA_GEOMETRY_H
