//
// Created by salmon on 16-11-29.
//

#include <simpla/parallel/MPIComm.h>
#include <simpla/third_part/SAMRAITimeIntegrator.h>
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

static bool _required_module_are_registered_ =                    //
    RegisterCreator<Tokamak>() &&                                 //
    RegisterCreator<mesh_type>() &&                               //
    RegisterCreator<engine::Domain<mesh_type, ICRFAntenna>>() &&  //
    RegisterCreator<engine::Domain<mesh_type, EMFluid>>() &&      //
    RegisterCreator<engine::Domain<mesh_type, PICBoris>>() &&     //
    RegisterCreator<engine::Domain<mesh_type, Maxwell>>();

}  // namespace simpla {

using namespace simpla;

int main(int argc, char** argv) {
    auto scenario = engine::Scenario::New();

    scenario->SetMesh(mesh_type::New());

    scenario->SetModel("Tokamak", engine::Model::New("lalala.gdsk"));

    scenario->NewSchedule<SAMRAITimeIntegrator>();

    //    scenario->template NewDomain<Maxwell>("Limiter");
    //    scenario->template NewDomain<EMFluid>("Plasma");

    scenario->Update();

    std::cout << *scenario << std::endl;

    //    scenario->Run();
}
