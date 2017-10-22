//
// Created by salmon on 17-2-21.
//
#include "GeoObject.h"

#include <memory>

#include "BoxUtilities.h"
#include "Cube.h"
namespace simpla {
namespace geometry {

GeoObject::GeoObject() = default;
GeoObject::~GeoObject() = default;

std::shared_ptr<data::DataNode> GeoObject::Serialize() const { return base_type::Serialize(); }
void GeoObject::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

box_type GeoObject::GetBoundingBox() const { return box_type{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}; }
// std::shared_ptr<GeoObject> GeoObject::GetBoundary() const { return nullptr; }

// Real GeoObject::Measure() const {
//    auto b = GetBoundingBox();
//    return (std::get<1>(b)[0] - std::get<0>(b)[0]) * (std::get<1>(b)[1] - std::get<0>(b)[1]) *
//           (std::get<1>(b)[2] - std::get<0>(b)[2]);
//};
//
// bool GeoObject::CheckInside(point_type const &x, Real tolerance) const {
//    return geometry::isInSide(GetBoundingBox(), x);
//}
// std::shared_ptr<GeoObject> GeoObject::Intersection(std::shared_ptr<GeoObject> const &other) const {
//    return Cube::New(geometry::Overlap(GetBoundingBox(), other->GetBoundingBox()));
//}
// std::shared_ptr<GeoObject> GeoObject::Difference(std::shared_ptr<GeoObject> const &other) const {
//    UNIMPLEMENTED;
//    return nullptr;
//}
// std::shared_ptr<GeoObject> GeoObject::Union(std::shared_ptr<GeoObject> const &other) const {
//    return Cube::New(geometry::Union(GetBoundingBox(), other->GetBoundingBox()));
//}
// Real GeoObject::isOverlapped(box_type const &b) const { return Measure(Overlap(GetBoundingBox(), b)) / measure(); }
//
// Real GeoObject::CheckOverlap(GeoObject const &other) const { return isOverlapped(other.GetBoundingBox()); }
//
// bool GeoObject::CheckInside(const point_type &x) const { return CheckInSide(GetBoundingBox(), x); };
//
// std::tuple<Real, point_type, point_type> GeoObject::ClosestPoint(point_type const &x) const {
//    return std::tuple<Real, point_type, point_type>{0, x, x};
//}

}  // namespace geometry {
}  // namespace simpla {