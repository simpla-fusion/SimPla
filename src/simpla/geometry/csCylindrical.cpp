//
// Created by salmon on 17-7-22.
//
#include "csCylindrical.h"
#include "Box.h"
#include "Circle.h"
#include "Curve.h"
#include "Cylinder.h"
#include "Line.h"
#include "Rectangle.h"
#include "Revolution.h"
namespace simpla {
namespace geometry {

csCylindrical::csCylindrical() : Chart() {}
csCylindrical::~csCylindrical() = default;
std::shared_ptr<simpla::data::DataNode> csCylindrical::Serialize() const { return base_type::Serialize(); };
void csCylindrical::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); };

std::shared_ptr<Curve> csCylindrical::GetAxis(index_tuple const &idx0, int dir) const {
    return GetAxis(local_coordinates(idx0), dir);
};
std::shared_ptr<GeoObject> csCylindrical::GetBoundingShape(box_type const &uvw) const {
    point_type u_min, u_max;
    std::tie(u_min, u_max) = std::get<1>(uvw) - std::get<0>(uvw);
    return Revolution::New(m_axis_, Face::New(b[0], b[1]), b[2]);
}

std::shared_ptr<Curve> csCylindrical::GetAxis(point_type const &uvw, int dir) const {
    std::shared_ptr<Curve> res = nullptr;

    switch (dir) {
        case PhiAxis:
            res = Circle::New(m_axis_, uvw[RAxis]);
            break;
        case ZAxis:
        case RAxis: {
            auto axis = m_axis_;
            //            axis.o[ZAxis] = xyz(uvw)[ZAxis];
            res = Line::New(axis.o, axis.GetDirection(dir), 1.0);
        } break;
        default:
            break;
    }
    return res;
}

}  // namespace geometry
}  // namespace simpla
