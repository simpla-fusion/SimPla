/** 
 * @file MeshWalker.h
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#ifndef SIMPLA_MESHWALKER_H
#define SIMPLA_MESHWALKER_H

#include <memory>
#include "../gtl/primitives.h"
#include "Mesh.h"

namespace simpla { namespace mesh
{

class MeshBase;

class MeshWalker
{
public:

    virtual std::shared_ptr<MeshWalker> clone(MeshBase const &) const = 0;

    virtual void update_ghost_from(MeshBase const &const &other) = 0;

    virtual bool check_mesh(MeshBase const &) const = 0;

    virtual std::vector<box_type> refine_boxes() const = 0;

    virtual void refine(MeshBase const &const &other) = 0;

    virtual bool coarsen(MeshBase const &const &other) = 0;

    virtual void work(Real dt) { }
};
}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESHWALKER_H
