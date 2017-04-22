//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/application/SpApp.h>
#include <simpla/data/all.h>
#include <simpla/engine/all.h>
#include <simpla/geometry/Cube.h>
#include <simpla/mesh/EBMesh.h>
#include <simpla/model/GEqdsk.h>
#include <iostream>
#include "simpla/mesh/CartesianGeometry.h"
#include "simpla/mesh/CylindricalGeometry.h"
#include "simpla/predefine/physics/EMFluid.h"
#include "simpla/predefine/physics/PEC.h"

namespace simpla {
namespace mesh {
REGISTER_CREATOR(engine::Chart, CartesianGeometry, " Cartesian Geometry")
REGISTER_CREATOR(engine::Chart, CylindricalGeometry, " Cylindrical Geometry")
}  // namespace mesh{
using namespace simpla::engine;
static bool _PRE_REGISTERED = EMFluid<CartesianRectMesh>::is_register &&                  //
                              EMFluid<CylindricalRectMesh>::is_register &&                //
                              EMFluid<mesh::EBMesh<CartesianRectMesh>>::is_register &&    //
                              EMFluid<mesh::EBMesh<CylindricalRectMesh>>::is_register &&  //
                              PEC<CartesianRectMesh>::is_register &&                      //
                              PEC<CylindricalRectMesh>::is_register &&                    //
                              PEC<mesh::EBMesh<CartesianRectMesh>>::is_register &&        //
                              PEC<mesh::EBMesh<CylindricalRectMesh>>::is_register;

struct UseCaseAMR : public application::SpApp {
    SP_OBJECT_HEAD(UseCaseAMR, application::SpApp)
    UseCaseAMR() = default;
    ~UseCaseAMR() override = default;
    SP_DEFAULT_CONSTRUCT(UseCaseAMR);

    void SetUp() override;
};
SP_REGISITER_APP(UseCaseAMR, " AMR Test ");

void UseCaseAMR::SetUp() {
    auto domain = std::make_shared<engine::Domain>();
    domain->SetGeoObject(std::make_shared<geometry::Cube>(box_type{{1, 0, 0.0}, {2, TWOPI, 2}}));
    domain->SetChart(engine::Chart::Create("CylindricalGeometry"));
    domain->GetChart()->SetOrigin(point_type{1, 0, 0});
    domain->GetChart()->SetDx(point_type{0.1, TWOPI / 64, 0.1});
    domain->SetWorker(engine::Worker::Create("EMFluid<CylindricalGeometry>"));
    domain->AddBoundaryCondition(engine::Worker::Create("PEC<CylindricalGeometry>"));

    auto ctx = std::make_shared<engine::Context>();
    ctx->GetAtlas().SetIndexBox(index_box_type{{0, 0, 0}, {64, 32, 64}});
    ctx->GetAtlas().SetPeriodicDimension(size_tuple{0, 0, 0});
    ctx->SetDomain("Center", domain);

    auto schedule = std::dynamic_pointer_cast<engine::TimeIntegrator>(engine::Schedule::Create("SAMRAI"));
    schedule->Initialize();
    schedule->SetTime(0.0);
    schedule->SetTimeStep(0.1);
    schedule->SetTimeEnd(1.0);
    schedule->SetContext(ctx);
    schedule->SetOutputURL("SimPLASaveData");
    schedule->SetUp();
    SetSchedule(schedule);
}

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