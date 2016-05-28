/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "MeshAtlas.h"
#include "MeshWorker.h"

void simpla::mesh::MeshAtlas::apply(mesh_id const &self, MeshWalker const &o_walker, Real dt)
{
    auto walker = o_walker.clone(*get(self));

    // update ghost from neighbours;
    for (auto const &neighbour:this->adjacent_blocks(self, 0)) { walker->update_ghost_from(*get(neighbour)); }
    walker->work(dt);


}

void simpla::mesh::MeshAtlas::update_level(int level, MeshWalker const &o_walker, Real dt)
{
    // 1. update remote data (MPI)

}

void simpla::mesh::MeshAtlas::apply(MeshWalker const &o_walker, Real dt)
{
    /** 1.|--|- global update ghost
     *    |  |- update center domain
     *  2.|--- update boundary domain
     *
     */
    this->apply(m_root_, o_walker, dt);


    auto children = this->adjacent_blocks(self, +1);
    // copy data to child box
    for (auto const &child:children) { walker->refine(*get(child)); }

    for (int n = 0; n < m_level_ratio_; ++n)
    {
        //parallel for
        for (auto const &child:children) { this->apply(child, *walker, dt / m_level_ratio_); }
        // TODO: add mpi sync at here
    }

    //copy data from lower level
    for (auto const &child:children) { if (walker->coarsen(*get(child))) { this->remove(child); }; }

}

std::vector<mesh_id>
simpla::mesh::MeshAtlas::adjacent_blocks(mesh_id const &id, int inc_level = 0)
{

}