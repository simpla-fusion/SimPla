//
// Created by salmon on 17-7-22.
//
#include "csCylindrical.h"
#include "Box.h"
#include "Cylinder.h"
#include "Edge.h"
#include "Face.h"
#include "Rectangle.h"
#include "Sweeping.h"
#include "gCircle.h"
#include "gCurve.h"
#include "gCylinder.h"
#include "gLine.h"
#include "gPlane.h"
#include "gSweeping.h"

namespace simpla {
namespace geometry {

csCylindrical::csCylindrical() = default;
csCylindrical::csCylindrical(csCylindrical const &other) = default;
csCylindrical::~csCylindrical() = default;
Axis csCylindrical::GetLocalAxis(point_type const &o) const {
    auto axis = m_axis_;
    axis.SetOrigin(xyz(o));
    axis.SetAxis(RAxis,
                 m_axis_.GetAxis(RAxis) * std::cos(o[PhiAxis]) + m_axis_.GetAxis(PhiAxis) * std::sin(o[PhiAxis]));
    axis.SetAxis(PhiAxis,
                 m_axis_.GetAxis(RAxis) * std::sin(o[PhiAxis]) - m_axis_.GetAxis(PhiAxis) * std::cos(o[PhiAxis]));
    return axis;
}

std::shared_ptr<Edge> csCylindrical::GetCoordinateEdge(point_type const &o, int normal, Real u) const {
    std::shared_ptr<gCurve> curve = nullptr;
    point_type axis_origin = (o);
    Real radius = o[RAxis];
    axis_origin[RAxis] = 0;
    auto axis = GetLocalAxis(axis_origin);
    switch (normal) {
        case PhiAxis:
            curve = gCircle::New(o[RAxis]);
            break;
        case ZAxis:
            curve = gLine::New(axis.GetAxis(ZAxis));
            break;
        case RAxis:
            curve = gLine::New(axis.GetAxis(RAxis));
            break;
        default:
            break;
    }
    return Edge::New(axis, curve, std::make_tuple(0, u));
}
std::shared_ptr<Face> csCylindrical::GetCoordinateFace(point_type const &o, int normal, Real u, Real v) const {
    std::shared_ptr<gSurface> surface = nullptr;
    point_type axis_origin = o;
    Real radius = o[RAxis];
    axis_origin[RAxis] = 0;
    auto axis = GetLocalAxis(axis_origin);
    switch (normal) {
        case PhiAxis: {
            surface = gPlane::New();
        } break;
        case ZAxis:
            surface = gDisk::New();
            break;
        case RAxis:
            surface = gCylindricalSurface::New();
            break;
        default:
            break;
    }
    return Face::New(axis, surface, std::make_tuple(point2d_type{0, u}, point2d_type{0, v}));
};

std::shared_ptr<GeoObject> csCylindrical::GetCoordinateBox(box_type const &b) const {
    point_type p0, p1;
    p0[0] = std::get<0>(b)[RAxis];
    p1[0] = std::get<1>(b)[RAxis];
    p0[1] = std::get<0>(b)[RAxis];
    p1[1] = std::get<1>(b)[ZAxis];
    p0[2] = std::get<0>(b)[PhiAxis];
    p1[2] = std::get<1>(b)[PhiAxis];
    auto axis = GetLocalAxis(std::get<0>(b));
    CHECK(p0);
    CHECK(p1);
    return GeoObjectHandle::New(
        axis, gSweeping::New(gPlane::New(), gCircle::New(std::get<0>(b)[RAxis]), Axis{{1, 0, 0}, {0, 0, 1}, {0, 1, 0}}),
        std::make_tuple(p0, p1));
};

}  // namespace geometry
}  // namespace simpla
