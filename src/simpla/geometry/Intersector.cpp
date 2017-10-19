//
// Created by salmon on 17-10-18.
//

#include "Intersector.h"
#include <simpla/data/DataNode.h>
#include <simpla/utilities/SPDefines.h>
#include "Body.h"
#include "Box.h"

namespace simpla {
namespace geometry {

Intersector::Intersector() = default;
Intersector::~Intersector() = default;

std::shared_ptr<Intersector> Intersector::New(std::shared_ptr<const GeoObject> const& geo, Real tolerance) {
    std::shared_ptr<Intersector> res = nullptr;
    if (auto g = std::dynamic_pointer_cast<const Body>(geo)) {}
    return res;
}
size_type Intersector::GetIntersectionPoints(std::shared_ptr<const Curve> const& curve,
                                             std::vector<Real>& intersection_point) const {
    size_type count = 0;

    return count;
}
// size_type Intersector::GetIntersectionCurve(std::shared_ptr<const Surface> const& curve,
//                                            std::vector<std::shared_ptr<Curve>>) {}
// size_type Intersector::GetIntersectionBody(std::shared_ptr<const Body> const& curve,
//                                           std::vector<std::shared_ptr<Body>>) {}

}  // namespace geometry {
}  // namespace simpla