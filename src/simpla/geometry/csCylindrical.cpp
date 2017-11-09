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

std::shared_ptr<Curve> csCylindrical::GetAxis(index_tuple const &idx0, int dir, index_type l) const {
    return GetAxis(local_coordinates(idx0), dir, static_cast<Real>(l));
};

std::shared_ptr<Curve> csCylindrical::GetAxis(point_type const &uvw, int dir, Real l) const {
    std::shared_ptr<Curve> res = nullptr;
    auto axis = m_axis_.Moved(xyz(uvw));
    switch (dir) {
        case PhiAxis:
            res = Circle::New(axis, uvw[RAxis], uvw[PhiAxis], l);
            break;
        case ZAxis:
        case RAxis:
            res = Line::New(axis.o, axis.o + axis.GetDirection(dir) * l);
            break;
        default:
            break;
    }
    return res;
}

}  // namespace geometry
}  // namespace simpla
