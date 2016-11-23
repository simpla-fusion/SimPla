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
struct GeometryBase : public MeshBlock
{

public:
    SP_OBJECT_HEAD(GeometryBase, MeshBlock)

    GeometryBase() {}

    template<typename ...Args>
    explicit GeometryBase(Args &&...args):MeshBlock(std::forward<Args>(args)...) {}

    virtual ~GeometryBase() {}

    virtual void initialize() { DO_NOTHING; }

    virtual void deploy() { DO_NOTHING; }
};
}}
#endif //SIMPLA_GEOMETRY_H
