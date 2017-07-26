//
// Created by salmon on 17-7-22.
//
#include "csCartesian.h"
#include "Curve.h"

#include <memory>

namespace simpla {
namespace geometry {
std::shared_ptr<Curve> csCartesian::GetAxisCurve(point_type const &x, int dir) const {
    vector_type v{0, 0, 0};
    v[dir] = 1;
    return std::shared_ptr<Curve>(new Line{x, v});
}
}  // namespace geometry
}  // namespace simpla
