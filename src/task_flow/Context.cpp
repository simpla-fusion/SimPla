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

void Context::next_step(Real dt)
{

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
//    // copy data to lower level
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
//    //copy data from lower level
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

