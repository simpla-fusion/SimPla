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

#include <simpla/model/Model.h>

#include <simpla/physics/Constants.h>
#include <simpla/simulation/TimeIntegrator.h>

//#include "amr_test.h"
#include "../../scenario/problem_domain/EMFluid.h"

using namespace simpla;


namespace simpla
{
std::shared_ptr<simulation::TimeIntegrator>
create_time_integrator(std::string const &name, std::shared_ptr<mesh::Worker> const &w);

std::shared_ptr<model::Model> create_model();

}//namespace simpla

int main(int argc, char **argv)
{
    logger::set_stdout_level(100);
    GLOBAL_COMM.init(argc, argv);
    // typedef mesh:: CylindricalGeometry mesh_type;
    // typedef AMRTest<mesh_type> work_type;
    index_box_type mesh_index_box{{0,   0,  0},
                                  {128, 32, 128}};
    auto worker = std::make_shared<EMFluid<mesh::CylindricalGeometry>>();

    auto model = create_model();

    model->db["global index box"] = mesh_index_box;

    model->set_chart(worker->get_chart());

    model->load(argv[1]);

    worker->set_model(model);

//    auto sp = w->add_particle("H", 1.0, 1.0);

    worker->print(std::cout);
    box_type bound_box{{1, 0,  -1},
                       {2, PI, 1}};
    // model->box();

    auto integrator = simpla::create_time_integrator("EMFluid", worker);

    integrator->db["CartesianGeometry"]["domain_boxes_0"] = mesh_index_box;
    integrator->db["CartesianGeometry"]["periodic_dimension"] = nTuple<int, 3>{1, 1, 1};
    integrator->db["CartesianGeometry"]["x_lo"] = std::get<0>(bound_box);
    integrator->db["CartesianGeometry"]["x_up"] = std::get<1>(bound_box);
    integrator->update();

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

    GLOBAL_COMM.close();

}

