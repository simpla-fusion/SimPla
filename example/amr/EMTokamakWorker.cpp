//
// Created by salmon on 16-11-29.
//



#include <simpla/SIMPLA_config.h>

#include <iostream>
#include <simpla/manifold/Atlas.h>
#include <simpla/manifold/Worker.h>
#include <simpla/manifold/Field.h>
#include <simpla/manifold/CartesianGeometry.h>
#include <simpla/manifold/CylindricalGeometry.h>
#include <simpla/model/GEqdsk.h>
#include <simpla/physics/Constants.h>
#include "../../scenario/problem_domain/EMFluid.h"

namespace simpla
{
class EMTokamakWorker;

std::shared_ptr<mesh::Worker> create_worker()
{
    return std::dynamic_pointer_cast<mesh::Worker>(std::make_shared<EMTokamakWorker>());
}


class EMTokamakWorker : public EMFluid<mesh::CylindricalGeometry>
{
    typedef EMFluid<mesh::CylindricalGeometry> base_type;
public:

    virtual void deploy();

    virtual void tear_down();

    virtual void update();

    virtual void initialize(Real data_time);

    virtual void set_physical_boundary_conditions(Real time);

    virtual void set_physical_boundary_conditions_E(Real time);

    virtual void set_physical_boundary_conditions_B(Real time);

    GEqdsk geqdsk;

    Bundle<Real, VERTEX, 9> m_volume_frac_{m_chart_, "m_volume_frac_", "INPUT"};
    Bundle<Real, VERTEX, 9> m_dual_volume_frac_{m_chart_, "m_dual_volume_frac_", "INPUT"};

    EntityIdRange edge_boundary;
    EntityIdRange face_boundary;
    EntityIdRange limiter_boundary;
    EntityIdRange vertex_boundary;
    EntityIdRange plasma_region_volume;
    EntityIdRange plasma_region_vertex;

    EntityIdRange J_src_range;
    std::function<Vec3(point_type const &, Real)> J_src_fun;

    EntityIdRange E_src_range;
    std::function<Vec3(point_type const &, Real)> E_src_fun;


};

void EMTokamakWorker::deploy()
{
    if (base_type::is_deployed()) { return; }
    base_type::deploy();


    // first run, only load configure, m_chart_=nullptr
    geqdsk.load(db["GEqdsk"].as<std::string>("geqdsk.gfile"));

    db["Particles"].foreach([&](std::string const &key, data::DataBase const &item) { add_particle(key, item); });

    db["bound box"] = geqdsk.box();
};

void EMTokamakWorker::tear_down() { base_type::tear_down(); };

void EMTokamakWorker::update() { base_type::update(); }

void EMTokamakWorker::initialize(Real data_time)
{
    base_type::initialize(data_time);


    for (auto &sp:particles())
    {
        sp.second->rho->assign([&](point_type const &x) { return std::sin(x[1]); });
    }

    auto m_start_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 3> *>(m_volume_frac_.data_block())->start();
    auto m_count_ = static_cast<mesh::DataBlockArray<Real, mesh::VERTEX, 3> *>(m_volume_frac_.data_block())->count();

    index_type ib = m_start_[0];
    index_type ie = m_start_[0] + m_count_[0];
    index_type jb = m_start_[1];
    index_type je = m_start_[1] + m_count_[1];
    index_type kb = m_start_[2];
    index_type ke = m_start_[2] + m_count_[2];

    field_type <VERTEX> rho0{m_chart_};
    rho0.clear();

    rho0.assign([&](point_type const &x)
                {
                    if (geqdsk.boundary().check_inside(x) > 0) { return geqdsk.profile("ne", x); }
                });

    for (auto &item:particles())
    {
        Real ratio = db["Particles"].at(item.first).get("ratio", 1.0);
        *item.second->rho = rho0 * ratio;
    }
//
//    for (index_type i = ib; i < ie; ++i)
//        for (index_type j = jb; j < je; ++j)
//            for (index_type k = kb; k < ke; ++k)
//            {
//                auto x = get_mesh()->point(i, j, k);
//            }


}

void EMTokamakWorker::set_physical_boundary_conditions(Real time)
{
    if (J_src_fun) { J1.assign([&](point_type const &x) { return J_src_fun(x, time); }, J_src_range); }
    if (E_src_fun) { E.assign([&](point_type const &x) { return E_src_fun(x, time); }, E_src_range); }
};

void EMTokamakWorker::set_physical_boundary_conditions_E(Real time)
{
    E.assign(0, edge_boundary);
}

void EMTokamakWorker::set_physical_boundary_conditions_B(Real time)
{
    B.assign(0, face_boundary);

}
}