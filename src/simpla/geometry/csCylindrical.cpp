//
// Created by salmon on 17-7-22.
//
#include "csCylindrical.h"
#include "Box.h"
#include "Curve.h"
#include "Cylinder.h"
#include "Edge.h"
#include "Face.h"
#include "gCylinder.h"
#include "gCircle.h"

namespace simpla {
namespace geometry {

csCylindrical::csCylindrical() = default;
csCylindrical::csCylindrical(csCylindrical const &other) = default;
csCylindrical::~csCylindrical() = default;

std::shared_ptr<Edge> csCylindrical::GetCoordinateEdge(point_type const &o, int normal, Real u) const {
    std::shared_ptr<Curve> curve = nullptr;
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
    std::shared_ptr<Face> res = nullptr;

    switch (normal) {
        case PhiAxis:
            res = gCylindricalSurface::New(o[RAxis]);
            break;
        case ZAxis: {
            res = Face::New(m_axis_, gDisk::New());

        } break;
        case RAxis: {
            res = Face::New(m_axis_,
                            gCylindricalSurface::New(o[RAxis], o[PhiAxis], o[PhiAxis] + u, o[ZAxis], o[ZAxis] + v));
        } break;
        default:
            break;
    }
    return res;
};

std::shared_ptr<Solid> csCylindrical::GetCoordinateBox(point_type const &o, Real u, Real v, Real w) const {
    return Cylinder::New(m_axis_, o, point_type{o[0] + u, o[1] + v, o[2] + w});
};

}  // namespace geometry
}  // namespace simpla
