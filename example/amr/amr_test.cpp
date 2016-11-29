//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>

#include <iostream>
#include <simpla/manifold/Worker.h>
#include <simpla/simulation/TimeIntegrator.h>
#include <simpla/physics/Constants.h>


using namespace simpla;


namespace simpla
{
std::shared_ptr<simulation::TimeIntegrator> create_time_integrator(std::string const &name);

std::shared_ptr<mesh::Worker> create_worker();


}//namespace simpla

int main(int argc, char **argv)
{
    logger::set_stdout_level(100);
    GLOBAL_COMM.init(argc, argv);
    // typedef mesh:: CylindricalGeometry mesh_type;
    // typedef AMRTest<mesh_type> work_type;

    auto worker = create_worker();

    worker->db["GEqdsk"] = std::string(argv[1]);
    worker->db["Particles"]["H"]["m"] = 1.0;
    worker->db["Particles"]["H"]["Z"] = 1.0;
    worker->db["Particles"]["H"]["ratio"] = 0.5;
    worker->db["Particles"]["D"]["m"] = 2.0;
    worker->db["Particles"]["D"]["Z"] = 1.0;
    worker->db["Particles"]["D"]["raito"] = 0.5;
    worker->db["Particles"]["e"]["m"] = SI_electron_proton_mass_ratio;
    worker->db["Particles"]["e"]["Z"] = -1.0;

    worker->deploy();

    worker->print(std::cout);

    index_box_type mesh_index_box{{0,  0,  0},
                                  {16, 16, 32}};

    auto bound_box = worker->db["bound_box"].as(box_type {{1, 0,  -1},
                                                          {2, PI, 1}});

    CHECK(bound_box);

    auto integrator = simpla::create_time_integrator("EMFluid");
    integrator->set_worker(worker);
    integrator->db["CartesianGeometry"]["domain_boxes_0"] = mesh_index_box;
    integrator->db["CartesianGeometry"]["periodic_dimension"] = nTuple<int, 3>{0, 1, 0};
    integrator->db["CartesianGeometry"]["x_lo"] = std::get<0>(bound_box);
    integrator->db["CartesianGeometry"]["x_up"] = std::get<1>(bound_box);

    integrator->deploy();

    integrator->check_point();


    INFORM << "***********************************************" << std::endl;

    while (integrator->remaining_steps())
    {
        integrator->next_step(0.01);
        integrator->check_point();
    }

    INFORM << "***********************************************" << std::endl;

    integrator.reset();

    INFORM << " DONE !" << std::endl;

    GLOBAL_COMM.close();

}

