//
// Created by salmon on 16-6-2.
//

#include <set>
#include "Select.h"

namespace simpla { namespace mesh
{

MeshEntityRange select(MeshBase const &m, MeshEntityRange const &r,
                       std::function<bool(point_type const &x)> const &pred)
{
    MeshEntityRange res;
//    std::set<MeshEntityId> i_set;
//
//    for (auto const &s:r) { if (pred(mesh.point(s))) { res.insert(s); }}

    return std::move(res);
}
}}//namespace simpla{namespace mesh{