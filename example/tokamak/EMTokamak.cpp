//
// Created by salmon on 16-11-29.
//

#include <simpla/parallel/MPIComm.h>
#include <simpla/predefine/engine/SimpleTimeIntegrator.h>
#include <simpla/third_part/SAMRAITimeIntegrator.h>
#include <simpla/utilities/Logo.h>
#include "simpla/application/SPInit.h"
#include "simpla/engine/Engine.h"
#include "simpla/engine/Mesh.h"
#include "simpla/geometry/csCylindrical.h"
#include "simpla/mesh/EBMesh.h"
#include "simpla/mesh/RectMesh.h"
#include "simpla/predefine/device/ICRFAntenna.h"
#include "simpla/predefine/device/Tokamak.h"
#include "simpla/predefine/physics/EMFluid.h"
#include "simpla/predefine/physics/Maxwell.h"
#include "simpla/predefine/physics/PICBoris.h"
#include "simpla/scheme/FVM.h"
namespace simpla {
typedef engine::Mesh<geometry::csCylindrical, mesh::RectMesh, mesh::EBMesh, scheme::FVM> mesh_type;
}  // namespace simpla {

using namespace simpla;
using namespace simpla::engine;

int main(int argc, char** argv) {
    simpla::Initialize(argc, argv);
    //    auto scenario = SAMRAITimeIntegrator::New();
    auto scenario = SimpleTimeIntegrator::New();

    scenario->SetMesh<mesh_type>();
    scenario->GetMesh()->GetChart()->SetScale({1, 1, 1});

    scenario->AddModel<Tokamak>("EAST", "/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900");
    scenario->SetDomain<Domain<mesh_type, Maxwell>>("EAST.Limiter");
    scenario->SetDomain<Domain<mesh_type, EMFluid>>("EAST.Plasma");

    scenario->SetTimeNow(0);
    scenario->SetTimeEnd(1.0);
    scenario->SetTimeStep(0.1);

    scenario->SetUp();

    TheStart();
    scenario->Run();
    TheEnd();
    std::cout << *scenario << std::endl;
}
