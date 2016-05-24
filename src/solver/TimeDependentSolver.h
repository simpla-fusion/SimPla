//
// Created by salmon on 16-5-23.
//

#ifndef SIMPLA_AMR_H
#define SIMPLA_AMR_H

#include "../mesh/MeshAtlas.h"

namespace simpla { namespace solver
{


class TimeDependentSolver
{

public:

    std::map<std::string, std::shared_ptr<mesh::MeshAttribute> > m_attributes_;

    TimeDependentSolver() { }

    virtual ~TimeDependentSolver() { }

    class Worker;

    class Mapper;

    bool check_amr() const { return m_enable_amr_; };

    void enable_amr() { m_enable_amr_ = true; }

    void disable_amr() { m_enable_amr_ = false; }

    void update_level(Real dt, int level = 0);


    virtual void map_to(mesh::MeshBlockId const &, mesh::MeshBlockId const &) = 0;

    virtual void time_step(mesh::MeshBlockId const &, Real dt) = 0;

    virtual bool need_coarsen(mesh::MeshBlockId const &) = 0;

    virtual std::vector<mesh::box_type> refine_boxes(int level = 0) = 0;


private:
    mesh::MeshAtlas m_atlas_;

    bool m_enable_amr_;

};

}}//namespace solver}//namespace simpla

#endif //SIMPLA_AMR_H
