//
// Created by salmon on 16-11-29.
//

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

static bool _required_module_are_registered_ =                                 //
    RegisterCreator<Tokamak>("Tokamak") &&                                     //
    RegisterCreator<mesh_type>("EBRectMesh") &&                                //
    RegisterCreator<engine::Domain<mesh_type, ICRFAntenna>>("ICRFAntenna") &&  //
    RegisterCreator<engine::Domain<mesh_type, EMFluid>>("EMFluid") &&          //
    RegisterCreator<engine::Domain<mesh_type, PICBoris>>("PICBoris") &&        //
    RegisterCreator<engine::Domain<mesh_type, Maxwell>>("Maxwell");

}  // namespace simpla {
