//
// Created by salmon on 17-11-19.
//

#include "Rectangle.h"
#include "Edge.h"
#include "Face.h"
#include "Rectangle.h"
#include "gPlane.h"

namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Rectangle)
Rectangle::Rectangle() = default;
Rectangle::Rectangle(Rectangle const &) = default;
Rectangle::~Rectangle() = default;
Rectangle::Rectangle(Axis const &axis, Real r0, Real r1, Real a0, Real a1)
    : Face(axis, gPlane::New(), r0, r1, a0, a1) {}
Rectangle::Rectangle(Axis const &axis, std::tuple<point2d_type, point2d_type> const &b)
    : Face(axis, gPlane::New(), b) {}
}  // namespace geometry {
}  // namespace simpla {