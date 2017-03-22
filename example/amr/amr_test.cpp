//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/engine/all.h>
#include <simpla/model/GEqdsk.h>
#include <simpla/parallel/MPIComm.h>
#include <simpla/physics/Constants.h>
#include <simpla/toolbox/FancyStream.h>
#include <iostream>

using namespace simpla;
int main(int argc, char **argv) {
    logger::set_stdout_level(100);
    GLOBAL_COMM.init(argc, argv);

    auto manager = std::make_shared<engine::Manager>();

    manager->db()->SetValue("name", "EMFluid");

    manager->db()->SetValue("CartesianGeometry/domain_boxes_0", {{0, 0, 0}, {64, 64, 64}});
    manager->db()->SetValue("CartesianGeometry/periodic_dimension", {0, 1, 0});

    {
        GEqdsk geqdsk;
        geqdsk.load(argv[1]);

        //        manager->GetModel().AddDomain("VACUUM", geqdsk.limiter_gobj());
        //        manager->GetModel().AddDomain("PLASMA", geqdsk.boundary_gobj());

        auto bound_box = manager->GetModel().bound_box();

        manager->db()->SetValue("CartesianGeometry/x_lo", std::get<0>(bound_box));
        manager->db()->SetValue("CartesianGeometry/x_up", std::get<1>(bound_box));
    }

    LOGGER << *manager->db() << std::endl;

    //    worker->db()->SetValue("Particles/H1/m", 1.0);
    //    worker->db()->SetValue("Particles/H1/Z", 1.0);
    //    worker->db()->SetValue("Particles/H1/ratio", 0.5);
    //    worker->db()->SetValue("Particles/D1/m", 2.0);
    //    worker->db()->SetValue("Particles/D1/Z", 1.0);
    //    worker->db()->SetValue("Particles/D1/ratio", 0.5);
    //    worker->db()->SetValue("Particles/e1/m", SI_electron_proton_mass_ratio);
    //    worker->db()->SetValue("Particles/e1/Z", -1.0);
    //    worker->db()->SetValue(
    //        "Particles", {"H"_ = {"m"_ = 1.0, "Z"_ = 1.0, "ratio"_ = 0.5}, "D"_ = {"m"_ = 2.0, "Z"_ = 1.0, "ratio"_ =
    //        0.5},
    //                      "e"_ = {"m"_ = SI_electron_proton_mass_ratio, "Z"_ = -1.0}});
    //
    //
    //
    //    manager->GetDomainView("PLASMA")->AddWorker(worker);

    manager->Initialize();
    INFORM << "***********************************************" << std::endl;

    //    while (manager->remainingSteps()) {
    //        manager->NextTimeStep(0.01);
    //        manager->CheckPoint();
    //    }

    INFORM << "***********************************************" << std::endl;

    manager.reset();

    INFORM << " DONE !" << std::endl;

    GLOBAL_COMM.close();
}
