//
// Created by salmon on 17-2-21.
//
#include "GeoObject.h"
#include "Cube.h"

namespace simpla {
namespace geometry {

box_type GeoObject::BoundingBox() const { return box_type{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}; }

Real GeoObject::Measure() const {
    auto b = BoundingBox();
    return (std::get<1>(b)[0] - std::get<0>(b)[0]) * (std::get<1>(b)[1] - std::get<0>(b)[1]) *
           (std::get<1>(b)[2] - std::get<0>(b)[2]);
};

bool GeoObject::CheckInside(point_type const &x) const { return CheckInSide(BoundingBox(), x); }

std::shared_ptr<data::DataTable> GeoObject::Serialize() const {
    auto res = engine::EnableCreateFromDataTable<GeoObject>::Serialize();
    return res;
}

void GeoObject::Deserialize(std::shared_ptr<data::DataTable> const &t) {
    engine::EnableCreateFromDataTable<GeoObject>::Deserialize(t);
}

// Real GeoObject::CheckOverlap(box_type const &b) const { return Measure(Overlap(BoundingBox(), b)) / measure(); }
//
// Real GeoObject::CheckOverlap(GeoObject const &other) const { return CheckOverlap(other.BoundingBox()); }
//
// bool GeoObject::CheckInside(const point_type &x) const { return CheckInSide(BoundingBox(), x); };
//
// std::tuple<Real, point_type, point_type> GeoObject::ClosestPoint(point_type const &x) const {
//    return std::tuple<Real, point_type, point_type>{0, x, x};
//}

}  // namespace geometry {
}  // namespace simpla {