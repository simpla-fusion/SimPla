/**
 * @file context.cpp
 *
 *  Created on: 2015-1-2
 *      Author: salmon
 */

#include "Context.h"

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>


namespace simpla
{

void Context::setup() { };

void Context::teardown() { };

io::IOStream &Context::check_point(io::IOStream &os) const
{
    return os;
}

io::IOStream &Context::save(io::IOStream &os) const
{
    return os;
}

io::IOStream &Context::load(io::IOStream &is)
{
    return is;
}

void Context::run(Real dt, int level)
{

    //TODO async run
    update(level); //  get data from parent level

    for (int i = 0; i < m_refine_ratio; ++i)
    {
        run(dt / m_refine_ratio, level + 1);
    }

    for (manifold::ChartBase const &chart_node: m_atlas_.find_at_level(level))
    {
        for (auto p_it = m_domains_.find(chart_node.id()); p_it != m_domains_.end(); ++p_it)
        {
            p_it->second->run(dt);
        };
    }

};

void Context::update(int level, int flag)
{

    //TODO async update

    for (manifold::ChartBase const &chart_node: m_atlas_.find_at_level(dest))
    {
        for (manifold::TransitionMap const &map_edge:m_atlas_.find_conection(chart_node.id()))
        {
            for (auto p_it = m_domains_.find(chart_node.id()); p_it != m_domains_.end(); ++p_it)
            {

                for (auto p_o_it = m_domains_.find(map_edge.second.id()); p_o_it != m_domains_.end(); ++p_o_it)
                {
                    p_it->second->sync(map_edge, p_o_it.second);
                }

            }
        };
    }


};



//void Context::apply(ProblemDomain &w, uuid const &id, Real dt)
//{
//
//    int ratio = m_mesh_atlas_.refine_ratio(id);
//
//    auto children = m_mesh_atlas_.children(id);
//
//    Real sub_dt = dt / ratio;
//
//    // copy m_data to lower level
//    for (auto &sub_id:children) { refine(id, sub_id); }
//
//    // push lower level
//    for (int n = 0; n < ratio; ++n)
//    {
//        for (auto &sub_id:children) { update(sub_id, sub_dt); }
//
//        for (auto &sub_id:children)
//        {
//            // move to lower level
//
//            for (auto const &oid:m_mesh_atlas_.neighbour(sub_id))
//            {
//                sync(sub_id, oid);
//            }
//        }
//
//        // TODO: add mpi sync at here
//    }
//
//    //copy m_data from lower level
//    for (auto &sub_id:children) { coarsen(sub_id, id); }
//    // push this level
//    update(id, id);
//}
//
//void Context::sync(get_mesh::uuid const &id, ProblemDomain w)
//{
//    w.view(id);
//
//    for (auto const &oid: m_mesh_atlas_.sibling(id)) { w.sync(oid); }
//}

}  // namespace simpla

