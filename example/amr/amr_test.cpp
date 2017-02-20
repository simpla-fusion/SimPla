//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTupleExt.h>
#include <simpla/data/DataTable.h>
#include <simpla/engine/DomainView.h>
#include <simpla/engine/Manager.h>
#include <simpla/engine/TimeIntegrator.h>
#include <simpla/engine/Worker.h>
#include <simpla/model/GEqdsk.h>
#include <simpla/parallel/MPIComm.h>
#include <simpla/physics/Constants.h>
#include <simpla/toolbox/FancyStream.h>
#include <iostream>
namespace simpla {
std::shared_ptr<engine::TimeIntegrator> create_time_integrator();
std::shared_ptr<engine::Worker> create_worker();
}  // namespace simpla
using namespace simpla;
using namespace simpla::data;
int main(int argc, char **argv) {
    logger::set_stdout_level(100);
    GLOBAL_COMM.init(argc, argv);

    auto manager = create_time_integrator();

    manager->db().SetValue("name", "EMFluid");
    manager->db().SetValue("CartesianGeometry.domain_boxes_0", index_box_type{{0, 0, 0}, {64, 64, 64}});
    manager->db().SetValue("CartesianGeometry.periodic_dimension", nTuple<int, 3>{0, 1, 0});

    {
        GEqdsk geqdsk;
        geqdsk.load(argv[1]);

        manager->GetModel().AddObject("VACUUM", geqdsk.limiter_gobj());
        manager->GetModel().AddObject("PLASMA", geqdsk.boundary_gobj());

        auto bound_box = manager->GetModel().bound_box();

        manager->db().SetValue("CartesianGeometry.x_lo", std::get<0>(bound_box));
        manager->db().SetValue("CartesianGeometry.x_up", std::get<1>(bound_box));
    }

    auto worker = create_worker();

    worker->db().SetValue("Particles.H.m", 1.0);
    worker->db().SetValue("Particles.H.Z", 1.0);
    worker->db().SetValue("Particles.H.ratio", 0.5);
    worker->db().SetValue("Particles.D.m", 2.0);
    worker->db().SetValue("Particles.D.Z", 1.0);
    worker->db().SetValue("Particles.D.ratio", 0.5);
    worker->db().SetValue("Particles.e.m", SI_electron_proton_mass_ratio);
    worker->db().SetValue("Particles.e.Z", -1.0);
    worker->db().SetValue("Particles"_ = {"H"_ = {"m"_ = 1.0, "Z"_ = 1.0, "ratio"_ = 0.5},
                                          "D"_ = {"m"_ = 2.0, "Z"_ = 1.0, "ratio"_ = 0.5},
                                          "e"_ = {"m"_ = SI_electron_proton_mass_ratio, "Z"_ = -1.0}});

    manager->GetDomainView("PLASMA").AppendWorker(worker);

    manager->Update();
    manager->CheckPoint();
    INFORM << "***********************************************" << std::endl;

    while (manager->remainingSteps()) {
        manager->NextTimeStep(0.01);
        manager->CheckPoint();
    }

    INFORM << "***********************************************" << std::endl;

    manager.reset();

    INFORM << " DONE !" << std::endl;

    GLOBAL_COMM.close();
}
