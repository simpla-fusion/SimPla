//
// Created by salmon on 17-10-20.
//

#include "Plane.h"
#include "GeoAlgorithm.h"
#include "Line.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Plane)

Plane::Plane() = default;
Plane::Plane(Plane const &) = default;
Plane::Plane(Axis const &axis) : base_type(axis) {}
Plane::Plane(point_type const &o, vector_type const &x, vector_type const &y) : Plane(Axis(o, x, y)) {}
Plane::~Plane() = default;

void Plane::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataEntry> Plane::Serialize() const { return base_type::Serialize(); }
// std::shared_ptr<PointsOnCurve> Plane::GetIntersection(std::shared_ptr<const Curve> const &g, Real tolerance) const {
//    std::shared_ptr<PointsOnCurve> res = nullptr;
//    if (auto line = std::dynamic_pointer_cast<const Line>(g)) {}
//    return res;
//}
//
// std::shared_ptr<Curve> Plane::GetIntersection(std::shared_ptr<const Surface> const &g, Real tolerance) const {
//    return nullptr;
//}
// bool Plane::CheckIntersection(point_type const &x, Real tolerance) const {
//    return std::abs(m_axis_.uvw(x)[2]) < tolerance;
//}
// bool Plane::CheckIntersection(box_type const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return false;
//}
}  // namespace geometry{
}  // namespace simpla{impla