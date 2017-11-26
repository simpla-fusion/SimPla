//
// Created by salmon on 17-11-26.
//

#ifndef SIMPLA_PREDEFINEDOMAINS_H
#define SIMPLA_PREDEFINEDOMAINS_H

#include <simpla/geometry/csCartesian.h>
#include <simpla/geometry/csCylindrical.h>
#include <simpla/mesh/CoRectMesh.h>
#include <simpla/mesh/RectMesh.h>
#include <simpla/scheme/FVM.h>
namespace simpla {
typedef engine::Domain<geometry::csCartesian, scheme::FVM, mesh::CoRectMesh> CartesianFVM;
typedef engine::Domain<geometry::csCylindrical, scheme::FVM, mesh::RectMesh> CylindricalFVM;
}  // namespace simpla {

#endif  // SIMPLA_PREDEFINEDOMAINS_H
