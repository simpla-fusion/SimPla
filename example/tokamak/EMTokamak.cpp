//
// Created by salmon on 16-11-29.
//
#include "simpla/SIMPLA_config.h"

#include <simpla/application/SPInit.h>
#include <simpla/engine/Engine.h>
#include <simpla/geometry/Cube.h>
#include <simpla/geometry/csCylindrical.h>
#include <simpla/mesh/EBMesh.h>
#include <simpla/mesh/RectMesh.h>
#include <simpla/parallel/MPIComm.h>
#include <simpla/predefine/device/ICRFAntenna.h>
#include <simpla/predefine/device/Tokamak.h>
#include <simpla/predefine/engine/SimpleTimeIntegrator.h>
#include <simpla/predefine/physics/EMFluid.h>
#include <simpla/predefine/physics/Maxwell.h>
#include <simpla/predefine/physics/PICBoris.h>
#include <simpla/scheme/FVM.h>
#include <simpla/third_part/SAMRAITimeIntegrator.h>
#include <simpla/utilities/Logo.h>
namespace simpla {
typedef engine::Domain<geometry::csCylindrical, scheme::FVM, mesh::RectMesh, mesh::EBMesh> domain_type;
}  // namespace simpla {

using namespace simpla;
using namespace simpla::engine;

int main(int argc, char** argv) {
    simpla::Initialize(argc, argv);
    //    auto scenario = SAMRAITimeIntegrator::New();
    auto scenario = SimpleTimeIntegrator::New();
    scenario->SetName("EAST");
    scenario->db()->SetValue("DumpFile", "EAST.xmf");
    scenario->GetAtlas()->SetChart<simpla::geometry::csCylindrical>();
    scenario->GetAtlas()->GetChart()->SetScale({0.1, 0.1, 0.1});

    auto tokamak = Tokamak::New("/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900");
    //    auto* p = new domain::Maxwell<domain_type>;
    scenario->SetDomain<domain::Maxwell<domain_type>>("Limiter", tokamak->Limiter());
    scenario->GetDomain("Limiter")->PreInitialCondition.Connect([=](DomainBase* self, Real time_now) {
        if (auto d = dynamic_cast<domain::Maxwell<domain_type>*>(self)) { d->B0v = tokamak->B0(); }
    });
    //    scenario->SetDomain<Domain<mesh_type, EMFluid>>("Plasma", tokamak->Boundary());
    //    scenario->GetDomain("Plasma")->PreInitialCondition.Connect([=](DomainBase* self, Real time_now) {
    //        if (auto d = dynamic_cast<Domain<mesh_type, EMFluid>*>(self)) { d->ne = tokamak->profile("ne"); }
    //    });
    scenario->SetTimeNow(0);
    scenario->SetTimeEnd(10.0);
    scenario->SetTimeStep(0.1);
    scenario->SetMaxStep(100);
    scenario->SetUp();
    std::cout << *scenario << std::endl;
    TheStart();
    scenario->Run();
    scenario->Dump();
    std::cout << *scenario << std::endl;
    TheEnd();
    scenario->TearDown();
}
