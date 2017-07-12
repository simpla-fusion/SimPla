//
// Created by salmon on 16-11-29.
//

#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
#include <simpla/mesh/EBMesh.h>
#include <simpla/mesh/RectMesh.h>
#include <simpla/predefine/device/ICRFAntenna.h>
#include <simpla/predefine/device/Tokamak.h>
#include <simpla/predefine/physics/EMFluid.h>
namespace simpla {

static bool _required_module_are_registered_ =
    Tokamak::is_registered &&  //
    engine::Domain<mesh::RectMesh, mesh::EBMesh, FVM, ICRFAntenna>::is_registered &&
    engine::Domain<mesh::RectMesh, mesh::EBMesh, FVM, EMFluid>::is_registered;

}  // namespace simpla {
