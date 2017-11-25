//
// Created by salmon on 17-11-17.
//

#include "Edge.h"
#include "gCurve.h"

namespace simpla {
namespace geometry {

Edge::Edge(Axis const &axis, std::shared_ptr<const gCurve> const &curve, std::tuple<Real, Real> const &b)
    : GeoObjectHandle(axis, curve, box_type{{std::get<0>(b), 0, 0}, {std::get<1>(b), 1, 1}}){};

void Edge::SetCurve(std::shared_ptr<const gCurve> const &s) {
    GeoObjectHandle::SetBasisGeometry(std::dynamic_pointer_cast<const gCurve>(s));
}
std::shared_ptr<const gCurve> Edge::GetCurve() const {
    return std::dynamic_pointer_cast<const gCurve>(GeoObjectHandle::GetBasisGeometry());
}

}  // namespace geometry
}  // namespace simpla