//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_CIRCLE_H
#define SIMPLA_CIRCLE_H
#include <simpla/utilities/Constants.h>
#include "Edge.h"
#include "Face.h"
namespace simpla {
namespace geometry {
struct Circle : public Edge {
    SP_GEO_OBJECT_HEAD(Edge, Circle)

   protected:
    explicit Circle(Axis const &axis, Real radius, Real a0, Real a1);
    explicit Circle(Axis const &axis, Real radius, Real a1 = TWOPI);
};
struct Disk : public Face {
    SP_GEO_OBJECT_HEAD(Face, Disk)

   protected:
    explicit Disk(Axis const &axis, Real r0, Real r1, Real a0, Real a1);
    explicit Disk(Axis const &axis, Real radius, Real a1 = TWOPI);
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_CIRCLE_H
