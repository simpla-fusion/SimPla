//
// Created by salmon on 17-7-22.
//
#include "csCartesian.h"
#include <memory>
#include "Box.h"
#include "Edge.h"
#include "Face.h"
#include "Line.h"
#include "Rectangle.h"
#include "Solid.h"
#include "gCurve.h"
namespace simpla {
namespace geometry {
csCartesian::csCartesian() = default;
csCartesian::csCartesian(csCartesian const &) = default;
csCartesian::~csCartesian() = default;

std::shared_ptr<Edge> csCartesian::GetCoordinateEdge(point_type const &o, int normal, Real u) const {
    return Line::New(m_axis_);
};
std::shared_ptr<Face> csCartesian::GetCoordinateFace(point_type const &o, int normal, Real u, Real v) const {
    return Rectangle::New(m_axis_, std::make_tuple(point2d_type{0.0, u}, point2d_type{0.0, v}));
};
std::shared_ptr<GeoObject> csCartesian::GetCoordinateBox(box_type const &b) const { return Box::New(b); }
}  // namespace geometry
}  // namespace simpla
