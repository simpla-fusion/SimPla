//
// Created by salmon on 17-7-22.
//
#include "csCylindrical.h"
#include "Box.h"
#include "Circle.h"
#include "Curve.h"
#include "Line.h"

namespace simpla {
namespace geometry {

csCylindrical::csCylindrical() : Chart() {}
csCylindrical::~csCylindrical() = default;
std::shared_ptr<simpla::data::DataNode> csCylindrical::Serialize() const { return base_type::Serialize(); };
void csCylindrical::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); };
std::shared_ptr<GeoObject> csCylindrical::GetBoundingShape(box_type const &b) const { return geometry::Box::New(b); };
std::shared_ptr<GeoObject> csCylindrical::GetBoundingShape(index_box_type const &b) const {
    return GetBoundingShape(
        std::make_tuple(local_coordinates(0, std::get<0>(b)), local_coordinates(0, std::get<0>(b))));
};
std::shared_ptr<Curve> csCylindrical::GetAxis(index_tuple const &idx0, int dir, index_type length) const {
    std::shared_ptr<Curve> res = nullptr;
    auto uvw = local_coordinates(idx0);
    vector_type R_axis{0, 0, 0};
    vector_type Z_axis{0, 0, 0};
    vector_type Phi_axis{0, 0, 0};
    R_axis[RAxis] = 1;
    Z_axis[ZAxis] = 1;
    Phi_axis[PhiAxis] = 1;
    switch (dir) {
        case PhiAxis: {
            point_type o = GetOrigin();
            o[ZAxis] = uvw[ZAxis];
            res =
                Circle::New(Axis{o, R_axis, Phi_axis, Z_axis}, uvw[RAxis], uvw[PhiAxis], length * GetScale()[PhiAxis]);
        } break;
        case ZAxis: {
            res = Line::New(uvw, uvw + Z_axis * length);
        } break;
        case RAxis: {
            res = Line::New(uvw, uvw + R_axis * length);
        } break;
        default:
            break;
    }
     return res;
};

std::shared_ptr<Curve> csCylindrical::GetAxis(point_type const &x0, const point_type &x1) const {
    point_type u = inv_map(x0);
    vector_type z_axis{0, 0, 1};
    vector_type r_axis{std::cos(u[PhiAxis]), std::sin(u[PhiAxis]), 0};
    std::shared_ptr<Curve> res = nullptr;
    FIXME;
    //    switch (x1 % 3) {
    //        case PhiAxis: {
    //            point_type o = {0, 0, x0[2]};
    //            //            res = Circle::New(o, u[RAxis], z_axis, r_axis);
    //        } break;
    //        case ZAxis: {
    //            res = Line::New(x0, x0 + z_axis);
    //        } break;
    //        case RAxis: {
    //            res = Line::New(x0, x0 + r_axis);
    //        } break;
    //        default:
    //            break;
    //    }

    return res;
}

box_type csCylindrical::GetBoundingBox(std::shared_ptr<geometry::GeoObject> const &geo) const {
    UNIMPLEMENTED;
    return box_type{{0, 0, 0}, {0, 0, 0}};
}

}  // namespace geometry
}  // namespace simpla
