//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>

#include <iostream>

#include <simpla/concept/Object.h>
#include <simpla/mesh/Atlas.h>
#include <simpla/geometry/CartesianGeometry.h>
#include <simpla/geometry/CylindricalGeometry.h>

#include <simpla/manifold/Field.h>
#include <simpla/physics/Constants.h>

#include <simpla/manifold/Calculus.h>
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

    typedef TM mesh_type;

    AMRTest() {}

    ~AMRTest() {}


    mesh::AttributeHolder m_attr_holder_;

    template<typename TV, mesh::MeshEntityType IFORM, size_type DOF = 1>
    using field_type=Field<TV, TM, index_const<IFORM>, index_const<DOF>>;

    Real epsilon = 1.0;
    Real mu = 1.0;

    field_type<Real, mesh::FACE> B{this, "B"};
    field_type<Real, mesh::EDGE> E{this, "E"};
    field_type<Real, mesh::EDGE> J{this, "J"};
    field_type<Real, mesh::VERTEX, 3> Ev{this, "Ev"};

    virtual std::shared_ptr<mesh::MeshBlock>
    create_mesh_block(int n, index_type const *lo,
                      index_type const *hi,
                      Real const *dx = nullptr,
                      Real const *x0 = nullptr,
                      id_type id = 0)
    {
        return std::dynamic_pointer_cast<mesh::MeshBlock>(std::make_shared<mesh_type>(n, lo, hi, dx, x0, id));
    };

    virtual void move_to(std::shared_ptr<mesh::MeshBlock> const &m_) { mesh::Worker::move_to(m_); }

    virtual mesh::AttributeHolder &attributes() { return m_attr_holder_; }

    virtual mesh::AttributeHolder const &attributes() const { return m_attr_holder_; }


    void initialize(Real data_time)
    {
        mesh_block()->connect(this);
        mesh_block()->initialize();
        Ev.clear();
        E.clear();
        B.clear();
        J.clear();
        E.foreach([&](point_type const &x)
                  {
                      return nTuple<Real, 3>{
                              std::sin((x[1])),
                              std::sin(TWOPI * (x[0] - 1)),
                              std::sin(TWOPI * (x[2]))
                      };
                  });
    }

    virtual void setPhysicalBoundaryConditions(double time)
    {

        auto b = mesh_block()->inner_index_box();
        index_tuple p = {NX / 2, NY / 2, NZ / 2};
        if (toolbox::is_inside(p, b)) { E(p[0], p[1], p[2], 0) = std::sin(omega * time); }

    };


    virtual void next_time_step(Real data_time, Real dt)
    {
        Ev = cross(Ev, Ev) * dot(Ev, Ev) * 2;
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

