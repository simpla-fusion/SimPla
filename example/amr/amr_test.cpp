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

    auto time_integrator = GLOBAL_TIME_INTEGRATOR_FACTORY.Create("samrai");

    auto ctx = time_integrator->GetContext();

    ctx->db()->SetValue("name", "EMTokamak");
    ctx->db()->SetValue("CartesianGeometry/domain_boxes_0", {{0, 0, 0}, {64, 64, 64}});
    ctx->db()->SetValue("CartesianGeometry/periodic_dimension", {0, 1, 0});
    {
        GEqdsk geqdsk;
        geqdsk.load(argv[1]);

        //        ctx->GetModel().AddDomain("VACUUM", geqdsk.limiter_gobj());
        //        ctx->GetModel().AddDomain("PLASMA", geqdsk.boundary_gobj());

        auto bound_box = ctx->GetModel().bound_box();

        ctx->db()->SetValue("CartesianGeometry/x_lo", std::get<0>(bound_box));
        ctx->db()->SetValue("CartesianGeometry/x_up", std::get<1>(bound_box));
    }

    LOGGER << *ctx->db() << std::endl;

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
    //    ctx->GetDomainView("PLASMA")->AddWorker(worker);

    time_integrator->Initialize();
    INFORM << "***********************************************" << std::endl;

    while (time_integrator->remainingSteps()) { time_integrator->NextTimeStep(0.01); }

    INFORM << "***********************************************" << std::endl;

    time_integrator->Finalize();

    INFORM << " DONE !" << std::endl;

    GLOBAL_COMM.close();
}
