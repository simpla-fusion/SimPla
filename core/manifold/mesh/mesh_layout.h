/**
 * @file mesh_layout.h
 * @author salmon
 * @date 2015-12-09.
 */

#ifndef SIMPLA_MESH_LAYOUT_H
#define SIMPLA_MESH_LAYOUT_H

#include <vector>
#include "mesh_block.h"

namespace simpla { namespace mesh
{
struct MeshLayout
{
    std::vector<MeshBlock> m_blocks_;
};

}}//namespace simpla { namespace mesh

#endif //SIMPLA_MESH_LAYOUT_H
