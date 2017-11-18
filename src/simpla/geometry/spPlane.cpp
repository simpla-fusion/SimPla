//
// Created by salmon on 17-10-20.
//

#include "spPlane.h"
#include "GeoAlgorithm.h"
#include "spLine.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(spPlane)

spPlane::spPlane() = default;
spPlane::spPlane(spPlane const &) = default;
spPlane::~spPlane() = default;

void spPlane::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataEntry> spPlane::Serialize() const { return base_type::Serialize(); }
// std::shared_ptr<PointsOnCurve> spPlane::GetIntersection(std::shared_ptr<const Curve> const &g, Real tolerance) const
// {
//    std::shared_ptr<PointsOnCurve> res = nullptr;
//    if (auto line = std::dynamic_pointer_cast<const spLine>(g)) {}
//    return res;
//}
//
// std::shared_ptr<Curve> spPlane::GetIntersection(std::shared_ptr<const Surface> const &g, Real tolerance) const {
//    return nullptr;
//}
// bool spPlane::CheckIntersection(point_type const &x, Real tolerance) const {
//    return std::abs(m_axis_.uvw(x)[2]) < tolerance;
//}
// bool spPlane::CheckIntersection(box_type const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return false;
//}
}  // namespace geometry{
}  // namespace simpla{impla