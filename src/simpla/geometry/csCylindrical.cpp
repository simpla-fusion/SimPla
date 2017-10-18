//
// Created by salmon on 17-7-22.
//
#include "csCylindrical.h"
#include "Cube.h"
#include "Curve.h"

namespace simpla {
namespace geometry {

csCylindrical::csCylindrical() : Chart() {}
csCylindrical::~csCylindrical() {}
std::shared_ptr<simpla::data::DataNode> csCylindrical::Serialize() const { return base_type::Serialize(); };
void csCylindrical::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); };
std::shared_ptr<GeoObject> csCylindrical::GetBoundingShape(box_type const &b) const { return geometry::Cube::New(b); };
std::shared_ptr<GeoObject> csCylindrical::GetBoundingShape(index_box_type const &b) const {
    return GetBoundingShape(
        std::make_tuple(local_coordinates(0, std::get<0>(b)), local_coordinates(0, std::get<0>(b))));
};

std::shared_ptr<Curve> csCylindrical::GetAxisCurve(point_type const &x, int dir) const {
    point_type u = inv_map(x);
    vector_type z_axis{0, 0, 1};
    vector_type r_axis{std::cos(u[PhiAxis]), std::sin(u[PhiAxis]), 0};
    std::shared_ptr<Curve> res = nullptr;
    switch (dir % 3) {
        case PhiAxis: {
            point_type o = {0, 0, x[2]};
            res = Circle::New(o, u[RAxis], z_axis, r_axis);
        } break;
        case ZAxis: {
            res = Line::New(x, x + z_axis);
        } break;
        case RAxis: {
            res = Line::New(x, x + r_axis);
        } break;
        default:
            break;
    }

    return res;
}

box_type csCylindrical::GetBoundingBox(std::shared_ptr<geometry::GeoObject> const &geo) const {
    UNIMPLEMENTED;
    return box_type{{0, 0, 0}, {0, 0, 0}};
}

}  // namespace geometry
}  // namespace simpla
