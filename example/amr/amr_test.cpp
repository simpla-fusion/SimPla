//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTupleExt.h>
#include <simpla/mesh/Worker.h>
#include <simpla/parallel/MPIComm.h>
#include <simpla/physics/Constants.h>
#include <simpla/simulation/TimeIntegrator.h>
#include <simpla/toolbox/FancyStream.h>
#include <iostream>

using namespace simpla;

namespace simpla {

using namespace data;

std::shared_ptr<simulation::TimeIntegrator> create_time_integrator(std::string const& str = "");

std::shared_ptr<mesh::Worker> create_worker();

std::shared_ptr<mesh::Mesh> create_mesh();
}  // namespace simpla

int main(int argc, char** argv) {
    logger::set_stdout_level(100);
    GLOBAL_COMM.init(argc, argv);
    // typedef mesh:: CylindricalGeometry mesh_type;
    // typedef AMRTest<mesh_type> work_type;

    auto worker = create_worker();

    //    worker->db.setValue("GEqdsk", argv[1]);
    //    worker->db.setValue("Particles.H.m", 1.0);
    //    worker->db.setValue("Particles.H.Z", 1.0);
    //    worker->db.setValue("Particles.H.ratio", 0.5);
    //    worker->db.setValue("Particles.D.m", 2.0);
    //    worker->db.setValue("Particles.D.Z", 1.0);
    //    worker->db.setValue("Particles.D.ratio", 0.5);
    //    worker->db.setValue("Particles.e.m", SI_electron_proton_mass_ratio);
    //    worker->db.setValue("Particles.e.Z", -1.0);

    worker->db.insert("GEqdsk"_ = argv[1],  //
                      "Particles"_ = {"H"_ = {"m"_ = 1.0, "Z"_ = 1.0, "ratio"_ = 0.5},
                                      "D"_ = {"m"_ = 2.0, "Z"_ = 1.0, "ratio"_ = 0.5},
                                      "e"_ = {"m"_ = SI_electron_proton_mass_ratio, "Z"_ = -1.0}});
    worker->Deploy();

    worker->Print(std::cout);

    auto bound_box = worker->db.getValue("bound_box", box_type{{1, 0, -1}, {2, PI, 1}});

    auto integrator = simpla::create_time_integrator();
    integrator->worker() = worker;
    integrator->db.setValue("name", "EMFluid");
    integrator->db.setValue("CartesianGeometry.domain_boxes_0", index_box_type{{0, 0, 0}, {64, 64, 64}});
    integrator->db.setValue("CartesianGeometry.periodic_dimension", nTuple<int, 3>{0, 1, 0});
    integrator->db.setValue("CartesianGeometry.x_lo", std::get<0>(bound_box));
    integrator->db.setValue("CartesianGeometry.x_up", std::get<1>(bound_box));

    integrator->Deploy();

    integrator->CheckPoint();

    INFORM << "***********************************************" << std::endl;

    while (integrator->remainingSteps()) {
        integrator->NextTimeStep(0.01);
        integrator->CheckPoint();
    }

    INFORM << "***********************************************" << std::endl;

    integrator.reset();

    INFORM << " DONE !" << std::endl;

    GLOBAL_COMM.close();
}
