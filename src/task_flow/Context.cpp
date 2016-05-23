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


namespace simpla { namespace task_flow
{
void Context::apply(Worker &w, uuid const &id, Real dt)
{
    w.view(id);

    int ratio = m_mesh_atlas_.refine_ratio(id);

    auto children = m_mesh_atlas_.children(id);

    Real sub_dt = dt / ratio;

    // copy data to lower level
    for (auto &sub_id:children) { w.refine(sub_id); }

    // push lower level
    for (int n = 0; n < ratio; ++n)
    {
        for (auto &sub_id:children) { apply(w, sub_id, sub_dt); }

        for (auto &sub_id:children)
        {
            // move to lower level
            w.view(sub_id);

            for (auto const &oid:m_mesh_atlas_.sibling(sub_id))
            {
                w.sync(oid);
            }
        }

        // TODO: add mpi sync at here
    }
    w.view(id);
    //copy data from lower level
    for (auto &sub_id:children) { w.coarsen(sub_id); }
    // push this level
    w.work(dt);
}

void Context::sync(mesh::uuid const &id, Worker w)
{
    w.view(id);

    for (auto const &oid: m_mesh_atlas_.sibling(id)) { w.sync(oid); }
}
}}  // namespace simpla

