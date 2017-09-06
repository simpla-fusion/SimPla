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
    scenario->SetName("EAST");
    scenario->SetMesh<mesh_type>();
    scenario->GetMesh()->GetChart()->SetScale({1, 1, 1});

    scenario->GetModel()->Load("gfile://home/salmon/workspace/SimPla/scripts/gfile/g038300.03900");
    scenario->SetDomain("Limiter", Domain<mesh_type, Maxwell>::New());
    scenario->SetDomain("Plasma", Domain<mesh_type, EMFluid>::New());

    scenario->SetTimeNow(0);
    scenario->SetTimeEnd(1.0);
    scenario->SetTimeStep(0.1);
    scenario->SetMaxStep(100);

    TheStart();
    scenario->SetUp();
    //    std::cout << *scenario << std::endl;

    scenario->Run();

    scenario->TearDown();

    TheEnd();
}
