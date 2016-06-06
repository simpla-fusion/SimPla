//
// Created by salmon on 16-6-2.
//

#include <set>
#include "../parallel/Parallel.h"
#include "MeshUtility.h"

namespace simpla { namespace mesh
{

//MeshEntityRange select(MeshBase const &m, MeshEntityRange const &r,
//                       std::function<bool(point_type const &x)> const &pred)
//{
//    parallel::concurrent_unordered_set<MeshEntityId> i_set;
//
//    for (auto const &s:r)
//    {
//        if (pred(m.point(s)))
//        {
//            i_set.insert(s);
//        }
//    }
//
//    return MeshEntityRange(i_set);
//}

MeshEntityRange select(MeshBase const &m, MeshEntityRange const &r, box_type const &b)
{
    parallel::concurrent_unordered_set<MeshEntityId> i_set;

    point_type xl, xu;
    std::tie(xl, xu) = b;

    for (auto const &s:r)
    {
        auto x0 = m.point(s);
        if (xl[0] <= x0[0] && x0[0] <= xu[0] &&
            xl[1] <= x0[1] && x0[1] <= xu[1] &&
            xl[2] <= x0[2] && x0[2] <= xu[2])
        {
            i_set.insert(s);
        }
    }


    return MeshEntityRange(i_set);
};

}}//namespace simpla{namespace get_mesh{