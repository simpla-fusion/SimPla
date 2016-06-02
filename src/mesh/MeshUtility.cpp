//
// Created by salmon on 16-6-2.
//

#include <set>
#include "MeshUtility.h"

namespace simpla { namespace mesh
{

MeshEntityRange select(MeshBase const &m, MeshEntityRange const &r,
                       std::function<bool(point_type const &x)> const &pred)
{
    auto res = MeshEntityRange::create<std::set<MeshEntityId>>();
    auto &i_set = res.as<std::set<MeshEntityId>>();

    for (auto const &s:r)
    {
        if (pred(m.point(s)))
        {
            i_set.insert(s);
        }
    }

    return res;
}
}}//namespace simpla{namespace mesh{