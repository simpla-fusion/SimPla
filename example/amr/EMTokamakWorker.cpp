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
public:
    SP_OBJECT_HEAD(EMTokamakWorker, EMFluid<mesh::CylindricalGeometry>);

    virtual void deploy();

    virtual void destroy();

    virtual void update();

    virtual void initialize(Real data_time);

    virtual void next_time_step(Real data_time, Real dt);

    virtual void set_physical_boundary_conditions(Real data_time);

    virtual void set_physical_boundary_conditions_E(Real time);

    virtual void set_physical_boundary_conditions_B(Real time);

    GEqdsk geqdsk;
    field_type <VERTEX> psi{m_chart_, "psi"};

//    Bundle<Real, VERTEX, 9> m_volume_frac_{m_chart_, "m_volume_frac_", "INPUT"};
//    Bundle<Real, VERTEX, 9> m_dual_volume_frac_{m_chart_, "m_dual_volume_frac_", "INPUT"};
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
    base_type::deploy();


    // first run, only load configure, m_chart_=nullptr
    geqdsk.load(db["GEqdsk"].as<std::string>("geqdsk.gfile"));

    db["Particles"].foreach([&](std::string const &key, data::DataBase const &item) { add_particle(key, item); });

    db["bound_box"] = geqdsk.box();
};

void EMTokamakWorker::destroy() { base_type::destroy(); };

void EMTokamakWorker::update() { base_type::update(); }

void EMTokamakWorker::initialize(Real data_time)
{
    base_type::initialize(data_time);

    rho0.assign([&](point_type const &x)
                {
                    return (geqdsk.in_boundary(x)) ? geqdsk.profile("ne", x) : 0.0;
                }
    );
    psi.assign([&](point_type const &x) { return geqdsk.psi(x); });
//
//    for (auto &item:particles())
//    {
//        Real ratio = db["Particles"].at(item.first).get("ratio", 1.0);
//        *item.second->rho = rho0 * ratio;
//    }
//
//    for (index_type i = ib; i < ie; ++i)
//        for (index_type j = jb; j < je; ++j)
//            for (index_type k = kb; k < ke; ++k)
//            {
//                auto x = get_mesh()->point(i, j, k);
//            }


}

void EMTokamakWorker::next_time_step(Real data_time, Real dt)
{

};

void EMTokamakWorker::set_physical_boundary_conditions(Real data_time)
{
    base_type::set_physical_boundary_conditions(data_time);
    if (J_src_fun) { J1.assign([&](point_type const &x) { return J_src_fun(x, data_time); }, J_src_range); }
    if (E_src_fun) { E.assign([&](point_type const &x) { return E_src_fun(x, data_time); }, E_src_range); }
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