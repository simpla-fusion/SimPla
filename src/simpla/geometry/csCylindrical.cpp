//
// Created by salmon on 17-7-22.
//
#include "csCylindrical.h"
#include "Box.h"
#include "gCurve.h"
#include "Cylinder.h"
#include "Edge.h"
#include "Face.h"
#include "gCircle.h"
#include "gCylinder.h"
#include "gPlane.h"

namespace simpla {
namespace geometry {

csCylindrical::csCylindrical() = default;
csCylindrical::csCylindrical(csCylindrical const &other) = default;
csCylindrical::~csCylindrical() = default;

std::shared_ptr<Edge> csCylindrical::GetCoordinateEdge(point_type const &o, int normal, Real u) const {
    std::shared_ptr<gCurve> curve = nullptr;
    //    switch (normal) {
    //        case PhiAxis:
    //            curve = gCircle::New(m_axis_, o[RAxis]);
    //            break;
    //        case ZAxis:
    //        case RAxis: {
    //            auto axis = m_axis_;
    //            //            axis.o[ZAxis] = xyz(o)[ZAxis];
    //            curve = spLine::New(axis.o, axis.GetDirection(normal), 1.0);
    //        } break;
    //        default:
    //            break;
    //    }
    return Edge::New(m_axis_, curve, 0, u);
}
std::shared_ptr<Face> csCylindrical::GetCoordinateFace(point_type const &o, int normal, Real u, Real v) const {
    std::shared_ptr<gSurface> surface = nullptr;

    switch (normal) {
        case PhiAxis: {
            auto cos_Phi = std::cos(o[PhiAxis]);
            auto sin_Phi = std::sin(o[PhiAxis]);
            surface = gPlane::New(m_axis_.x * cos_Phi + m_axis_.y * sin_Phi, m_axis_.x * sin_Phi - m_axis_.y * cos_Phi);
        } break;
        case ZAxis:
            surface = gDisk::New(o[RAxis]);
            break;
        case RAxis:
            surface = gCylindricalSurface::New(o[RAxis], o[PhiAxis], o[PhiAxis] + u, o[ZAxis], o[ZAxis] + v);
            break;
        default:
            break;
    }
    return Face::New(m_axis_, surface, o[(normal + 1) % 3], o[(normal + 1) % 3] + u, o[(normal + 2) % 3],
                     o[(normal + 2) % 3] + v);
};

std::shared_ptr<Solid> csCylindrical::GetCoordinateBox(point_type const &o, Real u, Real v, Real w) const {
    return Cylinder::New(m_axis_, o, point_type{o[0] + u, o[1] + v, o[2] + w});
};

}  // namespace geometry
}  // namespace simpla
