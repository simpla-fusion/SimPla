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

Cylinder::Cylinder(Axis const &axis, box_type const &b) : Solid(axis, gCylinder::New(), b) {}

CylinderSurface::CylinderSurface(Axis const &axis, Real radius, std::tuple<point2d_type, point2d_type> const &b)
    : Face(axis, gCylindricalSurface::New(radius), b) {}
}  // namespace geometry {
}  // namespace simpla {