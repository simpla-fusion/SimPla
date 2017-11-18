//
// Created by salmon on 17-11-6.
//

#include "Shape.h"
#include "Edge.h"
#include "Face.h"
#include "Solid.h"
namespace simpla {
namespace geometry {
Shape::Shape() = default;
Shape::Shape(Shape const &) = default;
Shape::~Shape() = default;
std::shared_ptr<Edge> Shape::MakeEdge(Axis const &axis, std::tuple<Real, Real> const &range) const {}
std::shared_ptr<Face> Shape::MakeFace(Axis const &axis, std::tuple<point2d_type, point2d_type> const &range) const {}
std::shared_ptr<Solid> Shape::MakeSolid(Axis const &axis, std::tuple<point_type, point_type> const &range) const {}
}  // namespace geometry{
}  // namespace simpla{