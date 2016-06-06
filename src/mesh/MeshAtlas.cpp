/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "MeshAtlas.h"

namespace simpla { namespace mesh
{

void MeshAtlas::decompose(int num, int rank)
{

};

void MeshAtlas::decompose(nTuple<int, 3> const &dims, nTuple<int, 3> const &self) { }

void MeshAtlas::load_balance() { }

void MeshAtlas::sync(std::string const &n, int level = 0) { }

void MeshAtlas::sync(int level = 0) { }

void MeshAtlas::refine(int level = 0) { }

void MeshAtlas::coarsen(int level = 0) { }

//void MeshAtlas::apply(MeshBlockId const &self, MeshWorker const &o_walker, Real dt)
//{
//    auto walker = o_walker.clone(*get(self));
//
//    // update ghost from neighbours;
//    for (auto const &neighbour:this->adjacent_blocks(self, 0)) { walker->update_ghost_from(*get(neighbour)); }
//    walker->work(dt);
//
//
//}
//
//void MeshAtlas::update_level(int level, MeshWorker const &o_walker, Real dt)
//{
//    // 1. update remote data (MPI)
//
//}
//
//void MeshAtlas::apply(MeshWorker const &o_walker, Real dt)
//{
//    /** 1.|--|- global update ghost
//     *    |  |- update center domain
//     *  2.|--- update boundary domain
//     *
//     */
//    this->apply(m_root_, o_walker, dt);
//
//
//    auto children = this->adjacent_blocks(self, +1);
//    // copy data to child box
//    for (auto const &child:children) { walker->refine(*get(child)); }
//
//    for (int n = 0; n < m_level_ratio_; ++n)
//    {
//        //parallel for
//        for (auto const &child:children) { this->apply(child, *walker, dt / m_level_ratio_); }
//        // TODO: add mpi sync at here
//    }
//
//    //copy data from lower level
//    for (auto const &child:children) { if (walker->coarsen(*get(child))) { this->remove(child); }; }
//
//}
//
//std::vector<MeshBlockId>
//MeshAtlas::adjacent_blocks(MeshBlockId const &id, int inc_level = 0)
//{
//
//}
}}//namespace simpla{namespace mesh{