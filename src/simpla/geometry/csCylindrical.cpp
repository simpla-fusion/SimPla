//
// Created by salmon on 17-7-22.
//
#include "csCylindrical.h"

namespace simpla {
namespace geometry {

std::shared_ptr<GeoObject> csCylindrical::BoundBox(box_type const &b) const { return nullptr; };
std::shared_ptr<GeoObject> csCylindrical::BoundBox(index_box_type const &b) const {
    return BoundBox(std::make_tuple(local_coordinates(std::get<0>(b)), local_coordinates(std::get<0>(b))));
};

std::shared_ptr<Curve> csCylindrical::GetAxisCurve(point_type const &x, int dir) const {
    vector_type z_axis{0, 0, 0}, r_axis{0, 0, 0};

    z_axis[ZAxis] = 1;
    r_axis[RAxis] = 1;
    Curve *res;
    switch (dir) {
        case PhiAxis:
            res = new Circle(GetOrigin(), x[RAxis], z_axis, r_axis);
            break;
        case ZAxis:
            res = new Line(x, z_axis);
            break;
        case RAxis:
        default:
            point_type v{0, 0, 0};
            v[(ZAxis + 0) % 3] = 0;
            v[(ZAxis + 1) % 3] = x[RAxis] * std::cos(x[PhiAxis]);
            v[(ZAxis + 2) % 3] = x[RAxis] * std::sin(x[PhiAxis]);
            res = new Line(x, v);
            break;
    }

    return std::shared_ptr<Curve>(res);
}
}  // namespace geometry
}  // namespace simpla
