/**
 * @file MeshAtlas.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHATLAS_H
#define SIMPLA_MESH_MESHATLAS_H

#include <boost/graph/adjacency_list.hpp>
#include "Mesh.h"
#include "MeshWalker.h"

namespace simpla { namespace mesh
{


class MeshAtlas
{

    uuid m_root_;
    int m_level_ratio_ = 2;
public:

    std::vector<uuid> find_neighbour(uuid const &id) const;

    std::vector<uuid> find_children(uuid const &id) const;

    int refine_ratio(uuid const &id);

    void add(std::vector<box_type> const &, int level = 0);

    int get_level(uuid const &) const;

    void apply(MeshWalker const &walker, Real dt);

    void apply(uuid const &, MeshWalker const &walker, Real dt);

    bool remove(uuid const &) const;

    MeshBase const *get(uuid) const;

};

}}//namespace simpla{namespace mesh{

#endif //SIMPLA_MESH_MESHATLAS_H
