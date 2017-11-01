//
// Created by salmon on 17-10-26.
//

#include "PolyPoints.h"
#include <utility>
namespace simpla {
namespace geometry {
PolyPoints::PolyPoints() = default;
PolyPoints::PolyPoints(PolyPoints const &) = default;
PolyPoints::PolyPoints(Axis const &axis) : GeoObject(axis){};

PolyPoints::~PolyPoints() = default;
std::shared_ptr<simpla::data::DataNode> PolyPoints::Serialize() const { return base_type::Serialize(); };
void PolyPoints::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); };
std::shared_ptr<GeoObject> PolyPoints::GetBoundary() const { return nullptr; }
box_type PolyPoints::GetBoundingBox() const {
    return std::make_tuple(point_type{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY},
                           point_type{SP_INFINITY, SP_INFINITY, SP_INFINITY});
}
bool PolyPoints::CheckIntersection(box_type const &, Real tolerance) const { return false; }
bool PolyPoints::CheckIntersection(point_type const &x, Real tolerance) const { return false; }
// bool PolyPoints::TestInsideUVW(point_type const &x, Real tolerance) const { return false; }
std::shared_ptr<GeoObject> PolyPoints::GetIntersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    return nullptr;
}

}  // namespace geometry
}  // namespace simpla