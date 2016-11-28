//
// Created by salmon on 16-11-25.
//

#ifndef SIMPLA_AMR_TEST_H
#define SIMPLA_AMR_TEST_H

#include <simpla/SIMPLA_config.h>

#include <iostream>

#include <simpla/concept/Object.h>

#include <simpla/manifold/Atlas.h>
#include <simpla/manifold/Worker.h>
#include <simpla/manifold/Chart.h>
#include <simpla/manifold/CartesianGeometry.h>
#include <simpla/manifold/CylindricalGeometry.h>
#include <simpla/manifold/Field.h>
#include <simpla/manifold/Calculus.h>

#include <simpla/physics/Constants.h>
#include <simpla/simulation/TimeIntegrator.h>


using namespace simpla;

template<typename TM>
struct AMRTest : public mesh::Worker
{
    SP_OBJECT_HEAD(AMRTest, mesh::Worker);


    AMRTest() {}

    ~AMRTest() {}

    mesh::Chart<TM> m_chart;

    template<typename TV, mesh::MeshEntityType IFORM, size_type DOF = 1>
    using field_type=Field<TV, TM, index_const<IFORM>, index_const<DOF>>;

    Real epsilon = 1.0;
    Real mu = 1.0;

    field_type<Real, mesh::FACE> B{&m_chart, "B"};
    field_type<Real, mesh::EDGE> E{&m_chart, "E"};
    field_type<Real, mesh::EDGE> J{&m_chart, "J"};
    field_type<Real, mesh::VERTEX, 3> Ev{&m_chart, "Ev"};
    field_type<Real, mesh::VERTEX, 3> Bv{&m_chart, "Bv"};
    field_type<Real, mesh::VERTEX, 3> Jv{&m_chart, "Jv"};


    virtual void move_to(std::shared_ptr<mesh::MeshBlock> const &m) { m_chart.move_to(m); }

    virtual mesh::ChartBase *chart() { return &m_chart; };

    virtual mesh::ChartBase const *chart() const { return &m_chart; };


    virtual void initialize(Real data_time = 0)
    {
        m_chart.initialize(data_time);
        Bv.clear();
        Ev.clear();
        Jv.clear();
        E.clear();
        B.clear();
        J.clear();

        Ev.assign([&](point_type const &x) { return x; });

        Bv.assign([&](point_type const &x) { return point_type{1, 0, 0}; });

    }

    virtual void set_physical_boundary_conditions(double time)
    {

//        index_tuple p = {NX / 2, NY / 2, NZ / 2};
//        if (m_chart.mesh()->is_inside(p)) { E(p[0], p[1], p[2], 0) = std::sin(omega * time); }
    };


    virtual void next_time_step(Real data_time, Real dt)
    {
//        Ev.update();
//        Bv.update();
        Jv = cross(Ev, Bv) * dot(Ev, Ev);//



//        E = E + (curl(B) / mu - J) * dt / epsilon;
//        B -= curl(E) * dt;

        E = map_to<mesh::EDGE>(Ev);
        B = map_to<mesh::FACE>(Bv);

//        Ev = map_to<mesh::VERTEX>(E);
//        Bv = map_to<mesh::VERTEX>(B);

    }


};

#endif //SIMPLA_AMR_TEST_H
