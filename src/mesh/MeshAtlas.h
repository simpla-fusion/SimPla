/**
 * @file MeshAtlas.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHATLAS_H
#define SIMPLA_MESH_MESHATLAS_H

#include <boost/graph/adjacency_list.hpp>
#include "Mesh.h"

namespace simpla { namespace mesh
{
class MeshBase;

class MeshAtlas
{
public:

    std::list<uuid> children(uuid const &id);

    std::list<uuid> sibling(uuid const &id);

    int refine_ratio(uuid const &id);

    MeshBase const *at(uuid const &id) const;
};

}}//namespace simpla{namespace mesh{

#endif //SIMPLA_MESH_MESHATLAS_H
