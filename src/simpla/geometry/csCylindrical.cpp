//
// Created by salmon on 17-7-22.
//
#include "csCylindrical.h"
#include "Box.h"
#include "Cylinder.h"
#include "Edge.h"
#include "Face.h"
#include "gCircle.h"
#include "gCurve.h"
#include "gCylinder.h"
#include "gLine.h"
#include "gPlane.h"

namespace simpla {
namespace geometry {

csCylindrical::csCylindrical() = default;
csCylindrical::csCylindrical(csCylindrical const &other) = default;
csCylindrical::~csCylindrical() = default;
Axis csCylindrical::GetLocalAxis(point_type const &o) const {
    auto axis = m_axis_;
    axis.SetOrigin(o);
    axis.SetAxis(RAxis,
                 m_axis_.GetAxis(RAxis) * std::cos(o[PhiAxis]) + m_axis_.GetAxis(PhiAxis) * std::sin(o[PhiAxis]));
    axis.SetAxis(PhiAxis,
                 m_axis_.GetAxis(RAxis) * std::sin(o[PhiAxis]) - m_axis_.GetAxis(PhiAxis) * std::cos(o[PhiAxis]));
    return axis;
}

std::shared_ptr<Edge> csCylindrical::GetCoordinateEdge(point_type const &o, int normal, Real u) const {
    std::shared_ptr<gCurve> curve = nullptr;
    point_type axis_origin = o;
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
    return Edge::New(axis, curve, 0, u);
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
            surface = gDisk::New(radius);
            break;
        case RAxis:
            surface = gCylindricalSurface::New(radius);
            break;
        default:
            break;
    }
    return Face::New(axis, surface, 0, u, 0, v);
};

std::shared_ptr<Solid> csCylindrical::GetCoordinateBox(point_type const &o, Real u, Real v, Real w) const {
    point_type axis_origin = o;
    Real radius = o[RAxis];
    axis_origin[RAxis] = 0;
    auto axis = GetLocalAxis(axis_origin);
    return Cylinder::New(axis, 0, u, 0, v, 0, w);
};

}  // namespace geometry
}  // namespace simpla
