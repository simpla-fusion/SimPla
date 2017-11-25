//
// Created by salmon on 17-11-19.
//

#include "Circle.h"
#include "Edge.h"
#include "Face.h"
#include "gCircle.h"
#include "gSurface.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Circle)
SP_GEO_OBJECT_REGISTER(Disk)
Circle::Circle(Axis const &axis, Real radius, std::tuple<Real, Real> const &r) : Edge(axis, gCircle::New(radius), r) {}
Circle::Circle(Axis const &axis, Real radius, Real a1) : Edge(axis, gCircle::New(radius), std::make_tuple(0, a1)) {}

Disk::Disk(Axis const &axis, std::tuple<point2d_type, point2d_type> const &b) : Face(axis, gDisk::New(), b) {}
Disk::Disk(Axis const &axis, Real radius, Real a1)
    : Disk(axis, std::make_tuple(point2d_type{0, radius}, point2d_type{0, a1})) {}
}  // namespace geometry {
}  // namespace simpla {