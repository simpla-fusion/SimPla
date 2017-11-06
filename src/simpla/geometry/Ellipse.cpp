//
// Created by salmon on 17-10-22.
//

#include "Ellipse.h"

namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Ellipse)
Ellipse::Ellipse() = default;
Ellipse::Ellipse(Ellipse const &other) = default;
Ellipse::Ellipse(Axis const &axis, Real major_radius, Real minor_radius)
    : Curve(axis), m_major_radius_(major_radius), m_minor_radius_(minor_radius) {}
Ellipse::~Ellipse() = default;

void Ellipse::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_major_radius_ = cfg->GetValue<Real>("MajorRadius", m_major_radius_);
    m_minor_radius_ = cfg->GetValue<Real>("MinorRadius", m_minor_radius_);
}
std::shared_ptr<simpla::data::DataNode> Ellipse::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("MajorRadius", m_major_radius_);
    res->SetValue<Real>("MinorRadius", m_minor_radius_);
    return res;
}
// bool Ellipse::CheckIntersection(point_type const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return false;
//}
// bool Ellipse::CheckIntersection(box_type const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return false;
//}
// std::shared_ptr<GeoObject> Ellipse::GetIntersectionion(std::shared_ptr<const GeoObject> const &, Real tolerance)
// const {
//    return nullptr;
//}
}  // namespace geometry{
}  // namespace simpla{