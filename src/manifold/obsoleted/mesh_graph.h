/**
 * @file mesh_graph.h
 * @author salmon
 * @date 2015-12-11.
 */

#ifndef SIMPLA_MESH_GRAPH_H
#define SIMPLA_MESH_GRAPH_H

#include "../../dataset/dataset.h"
#include "MeshBlock.h"

namespace simpla { namespace mesh
{

struct MeshMapEdge
{
    typedef MeshBlock m;

    typename m::id_type i0, i1;
    typename m::id_type m_offset_;
    typename m::box_type overlap_box;

    std::shared_ptr<m> m0, m1;


    virtual void update(DataSet *ds) = 0;

    template<typename TV, int IFORM> void apply(DataSet *ds)
    {
        MeshTransform<TV, IFORM>(*this).update(ds);
    };
};

template<typename TV, int IFORM>
struct MeshTransform : public MeshMapEdge
{
    typedef MeshIDs m;
    using MeshMapEdge::m0;
    using MeshMapEdge::m1;

    virtual void update(DataSet *ds);


};

struct MeshCoarsen : public MeshMapEdge
{
    virtual void update(DataSet *ds);
};

struct MeshRefinement : public MeshMapEdge
{
    virtual void update(DataSet *ds);
};

template<typename TV, int IFORM>
void MeshTransform<TV, IFORM>::update(DataSet *ds)
{
    DataSet &ds0;

    DataSet &ds1;


    m::id_type s_offset;

    parallel::parallel_for(
            m1.template make_range<IFORM>(overlap_box),
            [&](m::range_type const &r)
            {
                for (auto const &s1:r)
                {
                    m::id_type s0 = ((s1 & (~m::FULL_OVERFLOW_FLAG)) >> 1UL) + m_offset_;

                    ds1.template get_value<TV>(m1->hash(s1)) = m0->refinement<TV, IFORM>(s0, ds0);

                }
            }
    );
}


void MeshCoarsen::update(DataSet *ds)
{

//
//    parallel::parallel_for(
//            to_r,
//            [&](TRange const &r)
//            {
//                for (auto const &s1:to_r)
//                {
//                    m::id_type s0 = ((s1 & (~m::FULL_OVERFLOW_FLAG)) >> 1UL) + m_offset_;
//
//                    ds1.get_value(m1.hash(s1)) = m0.coarsen<TV, IFORM>(s0, ds0);
//
//                }
//            }
//    );
}

void MeshRefinement::update(DataSet *ds)
{

//    m::id_type s_offset;
//
//    parallel::parallel_for(
//            to_r,
//            [&](TRange const &r)
//            {
//                for (auto const &s1:to_r)
//                {
//                    m::id_type s0 = (s1 & (~m::FULL_OVERFLOW_FLAG)) + m_offset_;;
//
//                    ds1.get_value(m1.hash(s1)) = ds0.get_value<TV>(m0.hash(s0));
//
//                }
//            }
//    );
}

}}//namespace simpla { namespace mesh
#endif //SIMPLA_MESH_GRAPH_H
