//
// Created by salmon on 17-11-14.
//

#include "Face.h"
namespace simpla {
namespace geometry {

Face::Face(Axis const &axis, std::shared_ptr<const gSurface> const &surface,
           std::tuple<point2d_type, point2d_type> const &b)
    : GeoObjectHandle(axis, surface,
                      box_type{{std::get<0>(b)[0], std::get<0>(b)[1], 0}, {std::get<1>(b)[0], std::get<1>(b)[1], 1}}){};

void Face::SetSurface(std::shared_ptr<const gSurface> const &s) { GeoObjectHandle::SetBasisGeometry(s); }
std::shared_ptr<const gSurface> Face::GetSurface() const {
    return std::dynamic_pointer_cast<const gSurface>(GeoObjectHandle::GetBasisGeometry());
}

}  // namespace geometry
}  // namespace simpla