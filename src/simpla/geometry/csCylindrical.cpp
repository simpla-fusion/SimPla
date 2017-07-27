//
// Created by salmon on 17-7-22.
//
#include "csCylindrical.h"
#include "Curve.h"
namespace simpla {
namespace geometry {

std::shared_ptr<GeoObject> csCylindrical::BoundBox(box_type const &b) const { return nullptr; };
std::shared_ptr<GeoObject> csCylindrical::BoundBox(index_box_type const &b) const {
    return BoundBox(std::make_tuple(local_coordinates(std::get<0>(b)), local_coordinates(std::get<0>(b))));
};

std::shared_ptr<Curve> csCylindrical::GetAxisCurve(index_tuple const &idx, int dir) const {
    point_type u = local_coordinates(idx);
    point_type x = global_coordinates(idx);
    vector_type z_axis{0, 0, 1};
    vector_type r_axis{std::cos(u[PhiAxis]), std::sin(u[PhiAxis]), 0};
    Curve *res = nullptr;
    switch (dir%3) {
        case PhiAxis: {
            point_type o = {0, 0, x[2]};
            res = new Circle(o, u[RAxis], z_axis, r_axis);
        } break;
        case ZAxis: {
            res = new Line(x, z_axis);
        } break;
        case RAxis: {
            res = new Line(x, r_axis);
        } break;
        default:
            break;
    }

    return std::shared_ptr<Curve>(res);
}
}  // namespace geometry
}  // namespace simpla
