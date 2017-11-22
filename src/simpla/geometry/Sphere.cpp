//
// Created by salmon on 17-11-19.
//

#include "Sphere.h"
#include "Face.h"
#include "Solid.h"
#include "gSphere.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Sphere)
SP_GEO_OBJECT_REGISTER(SphereSurface)


Sphere::Sphere(Axis const &axis, Real r0, Real r1, Real a0, Real a1, Real h0, Real h1)
    : Solid(axis, simpla::geometry::gSphere::New(), r0, r1, a0, a1, h0, h1) {}


SphereSurface::SphereSurface(Axis const &axis, Real radius, Real a0, Real a1, Real h0, Real h1)
    : Face(axis, simpla::geometry::gSphereSurface::New(radius), a0, a1, h0, h1) {}
}  // namespace geometry {
}  // namespace simpla {