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

Sphere::Sphere(Axis const &axis, box_type const &range) : Solid(axis, simpla::geometry::gSphere::New(), range) {}

SphereSurface::SphereSurface(Axis const &axis, Real radius, std::tuple<point2d_type, point2d_type> const &b)
    : Face(axis, simpla::geometry::gSphereSurface::New(radius), b) {}
}  // namespace geometry {
}  // namespace simpla {