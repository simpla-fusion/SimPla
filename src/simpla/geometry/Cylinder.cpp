//
// Created by salmon on 17-7-22.
//

#include "Cylinder.h"
#include "ShapeBox.h"
#include "PointsOnCurve.h"

namespace simpla {
namespace geometry {
Cylinder::Cylinder() = default;
Cylinder::Cylinder(Cylinder const &) = default;
Cylinder::~Cylinder() = default;
Cylinder::Cylinder(Axis const &axis, Real radius, Real height, Real angle)
    : PrimitiveShape(axis), m_radius_(radius), m_height_(height), m_angle_(angle) {}
Cylinder::Cylinder(Real radius, Real height, Real angle)
    : PrimitiveShape(), m_radius_(radius), m_height_(height), m_angle_(angle) {}

std::shared_ptr<simpla::data::DataNode> Cylinder::Serialize() const { return base_type::Serialize(); };
void Cylinder::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

point_type Cylinder::xyz(Real r, Real angle, Real h) const {
    return m_axis_.xyz(r * std::cos(angle), r * std::sin(angle), h);
};
point_type Cylinder::uvw(Real x, Real y, Real z) const {
    auto xyz = m_axis_.uvw(x, y, z);
    return point_type{std::hypot(xyz[0], xyz[1]), std::atan2(xyz[1], xyz[0]), xyz[2]};
};
//
// bool Cylinder::CheckIntersection(point_type const &p, Real tolerance) const {
//    return m_shape_.Distance(m_axis_.uvw(p)) < 0;
//};
// bool Cylinder::CheckIntersection(box_type const &b, Real tolerance) const {
//    return m_shape_.TestBoxGetIntersectionion(m_axis_.uvw(std::get<0>(b)), m_axis_.uvw(std::get<1>(b)));
//};
// std::shared_ptr<Body> Cylinder::GetIntersection(std::shared_ptr<const Body> const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return nullptr;
//};
// std::shared_ptr<Curve> Cylinder::GetIntersection(std::shared_ptr<const Curve> const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return nullptr;
//};
// std::shared_ptr<Surface> Cylinder::GetIntersection(std::shared_ptr<const Surface> const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return nullptr;
//};
/**********************************************************************************************************************/

}  // namespace geometry {
}  // namespace simpla {
