//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>

#include <iostream>
#include <simpla/manifold/Atlas.h>
#include <simpla/manifold/Worker.h>
#include <simpla/manifold/Field.h>
#include <simpla/manifold/CartesianGeometry.h>
#include <simpla/manifold/CylindricalGeometry.h>

#include <simpla/physics/Constants.h>
#include <simpla/simulation/TimeIntegrator.h>

//#include "amr_test.h"
#include "../../scenario/problem_domain/EMFluid.h"

using namespace simpla;


namespace simpla
{
std::shared_ptr<simulation::TimeIntegrator>
create_time_integrator(std::string const &name, std::shared_ptr<mesh::Worker> const &w);
}//namespace simpla

int main(int argc, char **argv)
{
    logger::set_stdout_level(100);

    // typedef mesh::CartesianGeometry mesh_type;
//    typedef AMRTest<mesh_type> work_type;

    auto w = std::make_shared<EMFluid<mesh::CylindricalGeometry>>();

    auto sp = w->add_particle("H", 1.0, 1.0);


    w->print(std::cout);

    auto integrator = simpla::create_time_integrator("EMFluid", w);

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

