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
    typedef MeshIDs m;

    typedef m::id_type id_type;

    std::map<id_type, MeshBlock> m_blocks_;

};


template<typename M0, typename M1>
struct MeshMap
{
    typedef MeshIDs m;
    typedef typename MeshIDs::id_type id_type;


    M0 const &m0;
    M1 const &m1;
    id_type m_scale_, m_offset_;


    template<typename TV, int IFORM, typename TRange>
    void refinement(TRange const &to_r, DataSet &ds0, DataSet &ds1)
    {

        m::id_type s_offset;

        parallel::parallel_for(
                to_r,
                [&](TRange const &r)
                {
                    for (auto const &s1:r)
                    {
                        m::id_type s0 = ((s1 & (~m::FULL_OVERFLOW_FLAG)) >> 1UL) + m_offset_;

                        ds1.get_value(m1.hash(s1)) = m0.refinement<TV, IFORM>(s0, ds0);

                    }
                }
        );
    }

    template<typename TV, int IFORM, typename TRange>
    void coarsen(TRange const &to_r, DataSet &ds0, DataSet &ds1)
    {


        parallel::parallel_for(
                to_r,
                [&](TRange const &r)
                {
                    for (auto const &s1:to_r)
                    {
                        m::id_type s0 = ((s1 & (~m::FULL_OVERFLOW_FLAG)) >> 1UL) + m_offset_;

                        ds1.get_value(m1.hash(s1)) = m0.coarsen<TV, IFORM>(s0, ds0);

                    }
                }
        );
    }

    template<typename TV, int IFORM, typename TRange>
    void transform(TRange const &to_r, DataSet &ds0, DataSet &ds1)
    {

        m::id_type s_offset;

        parallel::parallel_for(
                to_r,
                [&](TRange const &r)
                {
                    for (auto const &s1:to_r)
                    {
                        m::id_type s0 = (s1 & (~m::FULL_OVERFLOW_FLAG)) + m_offset_;;

                        ds1.get_value(m1.hash(s1)) = ds0.get_value<TV>(m0.hash(s0));

                    }
                }
        );
    }
};


}}//namespace simpla { namespace mesh

#endif //SIMPLA_MESH_LAYOUT_H
