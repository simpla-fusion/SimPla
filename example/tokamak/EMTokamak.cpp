//
// Created by salmon on 16-11-29.
//


#include "simpla/engine/Engine.h"
#include "simpla/mesh/EBMesh.h"
#include "simpla/mesh/RectMesh.h"
#include "simpla/numeric/FVM.h"
#include "simpla/predefine/device/ICRFAntenna.h"
#include "simpla/predefine/device/Tokamak.h"
#include "simpla/predefine/physics/EMFluid.h"
namespace simpla {

static bool _required_module_are_registered_ =
    RegisterCreator<Tokamak>("Tokamak") &&
    RegisterCreator<engine::Domain<mesh::RectMesh, mesh::EBMesh, FVM, ICRFAntenna>>("ICRFAntenna") &&
    RegisterCreator<engine::Domain<mesh::RectMesh, mesh::EBMesh, FVM, EMFluid>>("EMFluid");

}  // namespace simpla {
