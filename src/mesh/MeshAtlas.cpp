/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "MeshAtlas.h"

void simpla::mesh::MeshAtlas::apply(uuid const &self, MeshWalker const &o_walker, Real dt)
{
    auto walker = o_walker.clone(self);

    for (auto const &neighbour:this->find_neighbour(self)) { walker->update_ghost_from(neighbour); }

    add(walker->refine_boxes(), get_level(self) + 1);

    auto children = this->find_children(self);
    // copy data to child box
    for (auto const &child:children) { walker->refine(child); }

    for (int n = 0; n < m_level_ratio_; ++n)
    {
        //parallel for
        for (auto const &child:children) { this->apply(child, *walker, dt / m_level_ratio_); }
        // TODO: add mpi sync at here
    }


    walker->work(dt);

    //copy data from lower level
    for (auto const &child:children) { if (walker->coarsen(child)) { this->remove(child); }; }


}

void simpla::mesh::MeshAtlas::apply(MeshWalker const &o_walker, Real dt)
{
    /** 1.|--|- global update ghost
     *    |  |- update center domain
     *  2.|--- update boundary domain
     *
     */
    apply(m_root_, o_walker, dt);
}