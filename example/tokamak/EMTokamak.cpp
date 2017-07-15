//
// Created by salmon on 16-11-29.
//

#include "simpla/engine/Engine.h"
#include "simpla/mesh/EBMesh.h"
#include "simpla/mesh/Maxwell.h"
#include "simpla/mesh/RectMesh.h"
#include "simpla/predefine/device/ICRFAntenna.h"
#include "simpla/predefine/device/Tokamak.h"
#include "simpla/predefine/physics/EMFluid.h"
#include "simpla/scheme/FVM.h"
namespace simpla {

static bool _required_module_are_registered_ =
    RegisterCreator<Tokamak>("Tokamak") &&
    RegisterCreator<engine::Domain<mesh::RectMesh, mesh::EBMesh, scheme::FVM, ICRFAntenna>>("ICRFAntenna") &&
    RegisterCreator<engine::Domain<mesh::RectMesh, mesh::EBMesh, scheme::FVM, EMFluid>>("EMFluid");
    RegisterCreator<engine::Domain<mesh::RectMesh, mesh::EBMesh, scheme::FVM, Maxwell>>("Maxwell");

}  // namespace simpla {
