//
// Created by salmon on 16-5-23.
//

#ifndef SIMPLA_AMR_H
#define SIMPLA_AMR_H

namespace simpla { namespace mesh
{


class AMR
{
    MeshAtlas m_atlas_;

    void apply(MeshWorker const &worker, Real dt) { apply(m_atlas_.root(), worker, dt); }

    void apply(uuid id, MeshWorker const &worker, Real dt);

};

}}//namespace mesh}//namespace simpla

#endif //SIMPLA_AMR_H
