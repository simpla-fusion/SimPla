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
Circle::Circle(Axis const &axis, Real radius, Real a0, Real a1) : Edge(axis, gCircle::New(radius), a0, a1) {}
Circle::Circle(Axis const &axis, Real radius, Real a1) : Edge(axis, gCircle::New(radius), 0, a1) {}

Disk::Disk(Axis const &axis, Real r0, Real r1, Real a0, Real a1) : Face(axis, gDisk::New(), r0, r1, a0, a1) {}
Disk::Disk(Axis const &axis, Real radius, Real a1) : Face(axis, gDisk::New(), 0, radius, 0, a1) {}
}  // namespace geometry {
}  // namespace simpla {