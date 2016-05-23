//
// Created by salmon on 16-5-23.
//
#include "TimeDependentSolver.h"

void simpla::solver::TimeDependentSolver::update_level(Real dt, int level)
{
    if (level == m_atlas_.max_level()) { return; }

#ifdef ENABLE_MPI
    for (auto const &m:m_atlas_.find(level, !mesh::LOCAL | mesh::ADJACENT))
    {
        UNIMPLEMENTED;
        //   receive  data  from remote block(MPI)
    }
    for (auto const &m:m_atlas_.find(level, mesh::LOCAL | mesh::ADJACENT))
    {
        //  send  data  to remote block(MPI)
        UNIMPLEMENTED;

    }
#endif// ENABLE_MPI

    for (auto const &self:m_atlas_.find(level, mesh::LOCAL))
    {
        // update ghost ;
        for (auto const &neighbour:m_atlas_.adjacent_blocks(self))
        {
            map_to(neighbour, self);
        }
    }

#ifdef ENABLE_AMR
    if (check_amr())
    {

        for (auto const &m:m_atlas_.find(level + 1, mesh::LOCAL))
        {
            if (need_coarsen(m)) { m_atlas_.remove(m); }
        }
        for (auto const &box:refine_boxes(level + 1))
        {
            m_atlas_.add(box, level + 1);
        }

        // push dt
        for (int n = 0; n < m_atlas_.level_ratio(); ++n)
        {
            update_level(level + 1, dt / static_cast<Real>(m_atlas_.level_ratio()));
        }
    }
#endif //ENABLE_AMR

    for (auto const &self:m_atlas_.find(level, mesh::LOCAL))
    {
        time_step(self, dt);
    }


}
