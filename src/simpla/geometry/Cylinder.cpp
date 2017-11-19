//
// Created by salmon on 17-11-19.
//

#include "Cylinder.h"
#include "Face.h"
#include "Solid.h"
#include "gCylinder.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Cylinder)
SP_GEO_OBJECT_REGISTER(CylinderSurface)
Cylinder::Cylinder() = default;
Cylinder::Cylinder(Cylinder const &) = default;
Cylinder::~Cylinder() = default;

Cylinder::Cylinder(Axis const &axis, Real r0, Real r1, Real a0, Real a1, Real h0, Real h1)
    : Solid(axis, gCylinder::New(), r0, r1, a0, a1, h0, h1) {}
Cylinder::Cylinder(Axis const &axis, box_type const &b) : Solid(axis, gCylinder::New(), b) {}

CylinderSurface::CylinderSurface() = default;
CylinderSurface::CylinderSurface(CylinderSurface const &) = default;
CylinderSurface::~CylinderSurface() = default;

CylinderSurface::CylinderSurface(Axis const &axis, Real radius, Real a0, Real a1, Real h0, Real h1)
    : Face(axis, gCylindricalSurface::New(radius), h0, h1, a0, a1) {}
}  // namespace geometry {
}  // namespace simpla {