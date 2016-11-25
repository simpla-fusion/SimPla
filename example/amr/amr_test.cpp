//
// Created by salmon on 16-11-4.
//

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

#define NX 64
#define NY 64
#define NZ 64
#define omega 1.0
using namespace simpla;


template<typename TM>
struct AMRTest : public mesh::Worker
{
    SP_OBJECT_HEAD(AMRTest, mesh::Worker);


    AMRTest()
    {

        for (auto const &item:m_chart.attributes())
        {
            std::cout << item->attribute()->name() << std::endl;
        }

    }

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


    virtual void initialize(Real data_time)
    {
        m_chart.initialize();
        Bv.clear();
        Ev.clear();
        Jv.clear();
        E.clear();
        B.clear();
        J.clear();

        Ev.assign([&](point_type const &x)
                  {

                      return nTuple<Real, 3>{x[0], 0, 0};
                  });

        Bv.assign([&](point_type const &x)
                  {

                      return nTuple<Real, 3>{0, x[0], 0};
                  });

    }

    virtual void set_physical_boundary_conditions(double time)
    {

        index_tuple p = {NX / 2, NY / 2, NZ / 2};
        if (m_chart.mesh()->is_inside(p)) { E(p[0], p[1], p[2], 0) = std::sin(omega * time); }
    };


    virtual void next_time_step(Real data_time, Real dt)
    {
        Ev.deploy();
        Bv.deploy();
        Jv = cross(Ev, Bv);//
        // ;
        //* dot(Ev, Ev) * 2;
//        E = E + (curl(B) / mu - J) * dt / epsilon;
//        B -= curl(E) * dt;
    }


};
namespace simpla
{
std::shared_ptr<simulation::TimeIntegrator>
create_time_integrator(std::string const &name, std::shared_ptr<mesh::Worker> const &w);
}//namespace simpla

int main(int argc, char **argv)
{
    logger::set_stdout_level(100);

    typedef mesh::CylindricalGeometry mesh_type;
    // typedef mesh::CartesianGeometry mesh_type;

    auto integrator = simpla::create_time_integrator("AMR_TEST", std::make_shared<AMRTest<mesh_type>>());
    integrator->deploy();
    integrator->check_point();

    INFORM << "***********************************************" << std::endl;

    while (integrator->remaining_steps())
    {
        integrator->next_step(0.01);
        integrator->check_point();
    }

    INFORM << "***********************************************" << std::endl;

    integrator->tear_down();

    integrator.reset();

    INFORM << " DONE !" << std::endl;
}

