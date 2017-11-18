//
// Created by salmon on 17-7-22.
//

#include "spCylinder.h"
#include "PointsOnCurve.h"

namespace simpla {
namespace geometry {
spCylinder::spCylinder() = default;
spCylinder::spCylinder(spCylinder const &) = default;
spCylinder::~spCylinder() = default;
spCylinder::spCylinder(Real radius, Real height) : m_radius_(radius), m_height_(height) {}
std::shared_ptr<simpla::data::DataEntry> spCylinder::Serialize() const { return base_type::Serialize(); };
void spCylinder::Deserialize(std::shared_ptr<const data::DataEntry> const &cfg) { base_type::Deserialize(cfg); }

// point_type spCylinder::xyz(Real r, Real angle, Real h) const {
//    return m_axis_.xyz(r * std::cos(angle), r * std::sin(angle), h);
//};
// point_type spCylinder::uvw(Real x, Real y, Real z) const {
//    auto xyz = m_axis_.uvw(x, y, z);
//    return point_type{std::hypot(xyz[0], xyz[1]), std::atan2(xyz[1], xyz[0]), xyz[2]};
//};
//
// bool spCylinder::CheckIntersection(point_type const &p, Real tolerance) const {
//    return m_shape_.Distance(m_axis_.uvw(p)) < 0;
//};
// bool spCylinder::CheckIntersection(box_type const &b, Real tolerance) const {
//    return m_shape_.TestBoxGetIntersectionion(m_axis_.uvw(std::get<0>(b)), m_axis_.uvw(std::get<1>(b)));
//};
// std::shared_ptr<Body> spCylinder::GetIntersection(std::shared_ptr<const Body> const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return nullptr;
//};
// std::shared_ptr<Curve> spCylinder::GetIntersection(std::shared_ptr<const Curve> const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return nullptr;
//};
// std::shared_ptr<Surface> spCylinder::GetIntersection(std::shared_ptr<const Surface> const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return nullptr;
//};
/**********************************************************************************************************************/

}  // namespace geometry {
}  // namespace simpla {
