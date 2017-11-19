//
// Created by salmon on 17-7-22.
//
#include "csCylindrical.h"
#include "Box.h"
#include "Curve.h"
#include "Edge.h"
#include "Face.h"
#include "Revolution.h"
#include "gCircle.h"
#include "gCylinder.h"
#include "gLine.h"
namespace simpla {
namespace geometry {

csCylindrical::csCylindrical() = default;
csCylindrical::csCylindrical(csCylindrical const &other) = default;
csCylindrical::~csCylindrical() = default;
std::shared_ptr<simpla::data::DataEntry> csCylindrical::Serialize() const { return base_type::Serialize(); };
void csCylindrical::Deserialize(std::shared_ptr<const data::DataEntry> const &cfg) { base_type::Deserialize(cfg); };

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
    return Edge::New(curve, 0, u);
}
std::shared_ptr<Face> csCylindrical::GetCoordinateFace(point_type const &o, int normal, Real u, Real v) const {
    std::shared_ptr<Surface> surface = nullptr;

    //    switch (normal) {
    //        case PhiAxis:
    //            surface = gCircle::New(m_axis_, uvw[RAxis]);
    //            break;
    //        case ZAxis:
    //        case RAxis: {
    //            auto axis = m_axis_;
    //            //            axis.o[ZAxis] = xyz(uvw)[ZAxis];
    //            surface = spLine::New(axis.o, axis.GetDirection(normal), 1.0);
    //        } break;
    //        default:
    //            break;
    //    }
    return Face::New(m_axis_, surface, 0, u, 0, v);
};

std::shared_ptr<Solid> csCylindrical::GetCoordinateBox(point_type const &o, Real u, Real v, Real w) const {
    return base_type::GetCoordinateBox(o, u, v, w);
};

}  // namespace geometry
}  // namespace simpla
