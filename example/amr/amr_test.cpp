//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/application/Application.h>
#include <simpla/data/all.h>
#include <simpla/engine/all.h>
#include <simpla/model/GEqdsk.h>
#include <simpla/parallel/MPIComm.h>
#include <simpla/physics/Constants.h>
#include <iostream>

using namespace simpla;
using namespace simpla::data;
namespace simpla {
SP_APP(AMR, "AMR Test") {
    auto time_integrator = std::dynamic_pointer_cast<engine::TimeIntegrator>(engine::Schedule::Create("SAMRAI"));
    time_integrator->Initialize();
    auto ctx = time_integrator->GetContext();

    time_integrator->Update();

    //    ctx->db()->SetValue("Domains", {"Center"_ = {"name"_ = "Center", "Mesh"_ = {"name"_ = "CartesianGeometry"},
    //                                                 "Worker"_ = {{"name"_ = "EMFluid"}}}});

    //    ctx->RegisterAttribute<int>("tag");
    //    ctx->RegisterAttribute<double, EDGE>("E");
    //    ctx->RegisterAttribute<double, FACE>("B");

    //    {
    //        GEqdsk geqdsk;
    //        geqdsk.load(argv[1]);
    //
    //        //        ctx->GetModel().AddDomain("VACUUM", geqdsk.limiter_gobj());
    //        //        ctx->GetModel().AddDomain("PLASMA", geqdsk.boundary_gobj());
    //
    //        auto bound_box = ctx->GetModel().bound_box();
    //    }
    //
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
    //    ctx->GetDomainView("PLASMA")->SetWorker(worker);
}
}  // namespace simpla{