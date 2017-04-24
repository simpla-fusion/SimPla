//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/application/SpApp.h>
#include <simpla/data/all.h>
#include <simpla/engine/all.h>
#include <simpla/geometry/Cube.h>
#include <simpla/model/GEqdsk.h>
#include <iostream>
#include "simpla/mesh/all.h"
#include "simpla/predefine/physics/EMFluid.h"
#include "simpla/predefine/physics/PEC.h"

namespace simpla {

namespace mesh {
REGISTER_CREATOR(CartesianGeometry)
REGISTER_CREATOR(CylindricalGeometry)
REGISTER_CREATOR(CartesianCoRectMesh)
REGISTER_CREATOR(CylindricalSMesh)
}

using namespace simpla::mesh;
using namespace simpla::engine;

static bool _PRE_REGISTERED = EMFluid<CartesianCoRectMesh>::is_registered &&  //
                              EMFluid<CylindricalSMesh>::is_registered &&     //
                              PEC<BoundaryMeshBase>::is_registered            //
    //  EMFluid<CartesianCoRectMeshEB>::is_register &&
    //  PEC<CartesianCoRectMeshEB>::is_register &&
    //  EMFluid<CylindricalSMeshEB>::is_register &&
    //  PEC<CylindricalSMeshEB>::is_register
    ;
struct UseCaseAMR : public application::SpApp {
    SP_OBJECT_HEAD(UseCaseAMR, application::SpApp)
    UseCaseAMR() = default;
    ~UseCaseAMR() override = default;
    SP_DEFAULT_CONSTRUCT(UseCaseAMR);
    DECLARE_REGISTER_NAME("UseCaseAMR")
    void SetUp() override;
};
REGISTER_CREATOR(UseCaseAMR);

void UseCaseAMR::SetUp() {
    //    CHECK(_PRE_REGISTERED);
    auto domain = std::make_shared<engine::Domain>();
    domain->SetGeoObject(std::make_shared<geometry::Cube>(box_type{{1.0, 0.0, 0.0}, {2, TWOPI, 2}}));
    domain->SetChart("CylindricalGeometry");
    domain->GetChart()->SetOrigin(point_type{1, 0, 0});
    domain->GetChart()->SetDx(point_type{0.1, TWOPI / 64, 0.1});
    domain->SetMesh("SMesh");
    domain->SetWorker("EMFluid");
    domain->AddBoundaryCondition("PEC");

    auto ctx = std::make_shared<engine::Context>();
    ctx->GetAtlas().SetIndexBox(index_box_type{{0, 0, 0}, {64, 32, 64}});
    ctx->GetAtlas().SetPeriodicDimension(size_tuple{0, 0, 0});
    ctx->SetDomain("Center", domain);

    auto schedule = std::dynamic_pointer_cast<engine::TimeIntegrator>(engine::Schedule::Create("SAMRAITimeIntegrator"));
    schedule->Initialize();
    schedule->SetTime(0.0);
    schedule->SetTimeStep(0.1);
    schedule->SetTimeEnd(1.0);
    schedule->SetContext(ctx);
    schedule->SetOutputURL("SimPLASaveData");
    schedule->SetUp();
    SetSchedule(schedule);
}

//    ctx->db()->SetValue("Domains", {"Center"_ = {"name"_ = "Center", "MeshBase"_ = {"name"_ = "CartesianGeometry"},
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
//        auto bound_box = ctx->GetModel().GetBoundBox();
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

}  // namespace simpla{