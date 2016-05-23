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
<<<<<<< HEAD:src/task_flow/Context.cpp
void Context::update(uuid id, Real dt)
=======
void Context::apply(Worker &w, uuid const &id, Real dt)
>>>>>>> 12337aa2f078bd60ab2d1de4b2ab4d77cc6d5beb:src/task_flow/Context.cpp
{

    int ratio = m_mesh_atlas_.refine_ratio(id);

    auto children = m_mesh_atlas_.children(id);

    Real sub_dt = dt / ratio;

    // copy data to lower level
    for (auto &sub_id:children) { refine(id, sub_id); }

    // push lower level
    for (int n = 0; n < ratio; ++n)
    {
        for (auto &sub_id:children) { update(sub_id, sub_dt); }

        for (auto &sub_id:children)
        {
            // move to lower level

            for (auto const &oid:m_mesh_atlas_.neighbour(sub_id))
            {
                sync(sub_id, oid);
            }
        }

        // TODO: add mpi sync at here
    }

    //copy data from lower level
    for (auto &sub_id:children) { coarsen(sub_id, id); }
    // push this level
    update(id, id);
}

void Context::sync(mesh::uuid const &id, Worker w)
{
    w.view(id);

    for (auto const &oid: m_mesh_atlas_.sibling(id)) { w.sync(oid); }
}
}}  // namespace simpla

