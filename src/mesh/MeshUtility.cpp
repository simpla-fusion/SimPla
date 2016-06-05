//
// Created by salmon on 16-6-2.
//

#include <set>
#include "../parallel/Parallel.h"
#include "MeshUtility.h"

namespace simpla { namespace mesh
{

MeshEntityRange select(MeshBase const &m, MeshEntityRange const &r,
                       std::function<bool(point_type const &x)> const &pred)
{
    parallel::concurrent_unordered_set<MeshEntityId> i_set;

    for (auto const &s:r)
    {
        if (pred(m.point(s)))
        {
            i_set.insert(s);
        }
    }

    return MeshEntityRange(i_set);
}
}}//namespace simpla{namespace mesh{