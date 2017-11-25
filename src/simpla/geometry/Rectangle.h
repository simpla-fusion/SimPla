//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_RECTANGLE_H
#define SIMPLA_RECTANGLE_H
#include "Face.h"
namespace simpla {
namespace geometry {

struct Rectangle : public Face {
    SP_GEO_OBJECT_HEAD(Face, Rectangle)
    explicit Rectangle(Axis const &axis, Real r0, Real r1, Real a0, Real a1);
    explicit Rectangle(Axis const &axis, std::tuple<point2d_type, point2d_type> const &);
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_RECTANGLE_H
