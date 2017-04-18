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
#include "simpla/mesh/CartesianGeometry.h"
#include "simpla/mesh/CylindricalGeometry.h"
#include "simpla/predefine/physics/EMFluid.h"
#include "simpla/predefine/physics/PEC.h"

namespace simpla {
namespace mesh {
REGISTER_CREATOR(engine::Chart, CartesianGeometry, " Cartesian Geometry")
REGISTER_CREATOR(engine::Chart, CylindricalGeometry, " Cylindrical Geometry")
}  // namespace mesh{

static bool _PRE_REGISTERED = EMFluid<mesh::CylindricalGeometry>::is_register &&
                              EMFluid<mesh::CartesianGeometry>::is_register &&
                              PEC<mesh::CylindricalGeometry>::is_register && PEC<mesh::CartesianGeometry>::is_register;

struct UseCaseAMR : public application::SpApp {
    UseCaseAMR();
    virtual ~UseCaseAMR();
    virtual std::shared_ptr<data::DataTable> Serialize() const;
    virtual void Deserialize(std::shared_ptr<data::DataTable>);
    virtual void Run();

   private:
    std::shared_ptr<engine::TimeIntegrator> m_schedule_;
};
SP_REGISITER_APP(UseCaseAMR, " AMR Test ");

UseCaseAMR::UseCaseAMR(){};

UseCaseAMR::~UseCaseAMR() { m_schedule_->Finalize(); }

void UseCaseAMR::Run() { m_schedule_->Run(); };

std::shared_ptr<data::DataTable> UseCaseAMR::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    if (m_schedule_ != nullptr) {
        res->Set("Schedule", m_schedule_->Serialize());
        res->SetValue("Schedule/Type", m_schedule_->GetClassName());
    }
    return res;
}
void UseCaseAMR::Deserialize(std::shared_ptr<data::DataTable> cfg) {
    m_schedule_ = std::dynamic_pointer_cast<engine::TimeIntegrator>(engine::Schedule::Create("SAMRAI"));
    m_schedule_->Initialize();
    m_schedule_->SetOutputURL(cfg->GetValue<std::string>("output", "SimPLASaveData"));
    if (cfg->GetTable("Schedule") == nullptr) {
        auto domain = m_schedule_->GetContext()->GetDomain("Center");
        domain->SetGeoObject(std::make_shared<geometry::Cube>(box_type{{-1, 2, 0.0}, {12, 13, 14}}));
        domain->SetChart(engine::Chart::Create("CylindricalGeometry"));
        domain->GetChart()->SetOrigin(point_type{1, 1, 1});
        domain->SetWorker(engine::Worker::Create("EMFluid<CylindricalGeometry>"));
        domain->AddBoundaryCondition(engine::Worker::Create("PEC<CylindricalGeometry>"));
        m_schedule_->SetTime(0.0);
        m_schedule_->SetTimeStep(0.1);
        m_schedule_->SetTimeEnd(1.0);
    } else {
        m_schedule_->Deserialize(cfg->GetTable("Schedule"));
    }
    m_schedule_->SetUp();
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