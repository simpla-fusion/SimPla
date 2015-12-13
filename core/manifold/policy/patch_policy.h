/**
 * @file patch.h
 * @author salmon
 * @date 2015-11-19.
 */

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include "../../parallel/parallel.h"
#include "../../dataset/dataset.h"
#include "../../geometry/cut_cell.h"

namespace simpla { namespace manifold { namespace policy
{

template<typename ...> struct PatchPolicy;

template<typename TMesh>
struct PatchPolicy<TMesh>
{

private:
    typedef TMesh mesh_type;

    typedef typename mesh_type::range_type range_type;

    typedef PatchPolicy<mesh_type> this_type;

    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::box_type box_type;

public:

    typedef this_type patch_policy;


    virtual ~PatchPolicy() { }


}; //template<typename TMesh> struct ParallelPolicy

//
//template<typename TMesh>
//template<typename TV, int IFORM, typename TRange>
//bool PatchPolicy<TMesh>::map_to(mesh_type const &m0, DataSet const &ds0, mesh_type const &m1, DataSet &ds1)
//{
//
//    box_type box = m1.out_box();
//
//    if (!intersection(m0.box(), &box)) { return false; }
//
//
//    auto r0 = m1.template make_range<IFORM>(box);
//
//    id_type shift = 0UL;// fixme shift should not be 0
//
//    if (m1.level == m0.level)
//    {
//        parallel::parallel_for(r0, [&](range_type const &r)
//        {
//            for (auto const &s:r)
//            {
//                ds1.template get_value<TV>(m1.hash(s)) =
//                        ds1.template get_value<TV>(m0.hash((s & (~mesh_type::FULL_OVERFLOW_FLAG)) + shift));
//            }
//        }
//        );
//
//    }
//    else if (m1.level < m0.level)
//    {
//        parallel::parallel_for(r0, [&](range_type const &r)
//        {
//            for (auto const &s:r)
//            {
//                ds1.template get_value<TV>(m1.hash(s)) =
//                        coarsen<TV, IFORM>(m0, ds0, (s & (~mesh_type::FULL_OVERFLOW_FLAG)) + shift);
//            }
//        }
//        );
//
//    }
//    else if (m1.level > m0.level)
//    {
//        parallel::parallel_for(r0, [&](range_type const &r)
//        {
//            for (auto const &s:r)
//            {
//                ds1.template get_value<TV>(m1.hash(s)) =
//                        refinement<TV, IFORM>(m0, ds0, (s & (~mesh_type::FULL_OVERFLOW_FLAG)) + shift);
//            }
//        }
//        );
//
//    }
//
//
//};
//
//
//template<typename TMesh> template<typename TV, int IFORM>
//TV PatchPolicy<TMesh>::coarsen(mesh_type const &m, DataSet const &ds, id_type const &s) const
//{
//    return (ds.get_value<TV>(m.hash(s + H)) + ds.get_value<TV>(m.hash(s + L))) * 0.5;
//}
//
//template<typename TMesh> template<typename TV, int IFORM>
//TV PatchPolicy<TMesh>::refinement(mesh_type const &m, DataSet const &ds, id_type const &s) const
//{
//    return (ds.get_value<TV>(m.hash(s + H)) + ds.get_value<TV>(m.hash(s + L))) * 0.5;
//}


}}}//namespace simpla { namespace manifold { namespace policy

#endif //SIMPLA_PATCH_H
