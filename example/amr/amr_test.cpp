//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>

#include <iostream>
#include <simpla/manifold/pre_define/PreDefine.h>

#include <simpla/concept/Object.h>
#include <simpla/mesh/Atlas.h>
#include <simpla/geometry/CartesianGeometry.h>
#include <simpla/physics/Field.h>
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

    typedef typename TM::mesh_type mesh_type;

    AMRTest() {}

    ~AMRTest() {}

    TM m;

    template<typename TV, mesh::MeshEntityType IFORM, size_type DOF = 1>
    using field_type=Field<TV, TM, index_const<IFORM>, index_const<DOF>>;

    Real epsilon = 1.0;
    Real mu = 1.0;

    field_type<Real, mesh::FACE> B{m, "B"};
    field_type<Real, mesh::EDGE> E{m, "E"};
    field_type<Real, mesh::EDGE> J{m, "J"};

    mesh::MeshBlock const *mesh() const { return mesh_block(); }

    virtual void move_to(std::shared_ptr<mesh::MeshBlock> const &m_)
    {
        m.move_to(m_);
        mesh::Worker::move_to(m_);
    }

    virtual void foreach(std::function<void(mesh::AttributeViewBase const &)> const &fun) const { m.foreach(fun); };

    virtual void foreach(std::function<void(mesh::AttributeViewBase &)> const &fun) { m.foreach(fun); };


    virtual std::shared_ptr<mesh::MeshBlock>
    create_mesh_block(index_type const *lo, index_type const *hi, Real const *dx,
                      Real const *xlo = nullptr, Real const *xhi = nullptr) const
    {
        auto res = std::make_shared<mesh::MeshBlock>(3, lo, hi, dx, xlo, xhi);
        res->deploy();
        return res;
    };


    void initialize(Real data_time)
    {
        m.deploy();
        E.clear();
        B.clear();
        J.clear();
        E.foreach([&](point_type const &x)
                  {
                      return nTuple<Real, 3>{
                              std::sin(x[0] * TWOPI / 16),//* std::sin(x[1] * m_k_[1]) * std::sin(x[2] * m_k_[2]),
                              0,//  std::cos(x[0] * m_k_[0]) * std::cos(x[1] * m_k_[1]) * std::cos(x[2] * m_k_[2]),
                              0//  std::sin(x[0] * m_k_[0]) * std::cos(x[1] * m_k_[1]) * std::sin(x[2] * m_k_[2])
                      };
                  });
//        xyz.clear();


    }

    virtual void setPhysicalBoundaryConditions(double time)
    {

        auto b = mesh()->inner_index_box();

        index_tuple p = {NX / 2, NY / 2, NZ / 2};

        if (toolbox::is_inside(p, b)) { E(p[0], p[1], p[2], 0) = std::sin(omega * time); }

    };


    virtual void next_time_step(Real data_time, Real dt)
    {
        VERBOSE << "box = " << mesh()->dx() << " time = " << std::setw(12) << std::left << data_time << " dt ="
                << std::setw(12) << dt << std::endl;
//        E = E + (curl(B) / mu - J) * dt / epsilon;
        B -= curl(E) * dt;
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

//    typedef manifold::CylindricalManifold mesh_type;
    typedef manifold::CartesianManifold mesh_type;

    auto integrator = simpla::create_time_integrator("AMR_TEST", std::make_shared<AMRTest<mesh_type>>());

    integrator->deploy();

    integrator->check_point();

    INFORM << "***********************************************" << std::endl;

    while (integrator->remaining_steps())
    {
        integrator->next_step(1);
        integrator->check_point();
    }

    INFORM << "***********************************************" << std::endl;

    integrator->tear_down();

    integrator.reset();

    INFORM << " DONE !" << std::endl;
}

