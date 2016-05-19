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

class MeshAtlas
{
    MeshBase root_;

    boost::adjacency_list<> m_atlas_;

    void add_edge()
    {
        boost::add_edge(u, v);
    }

};

}}//namespace simpla{namespace mesh{

#endif //SIMPLA_MESH_MESHATLAS_H
