//
// Created by salmon on 17-10-22.
//

#include "spEllipse.h"

namespace simpla {
namespace geometry {
SP_SHAPE_REGISTER(spEllipse)
spEllipse::spEllipse() = default;
spEllipse::spEllipse(spEllipse const &other) = default;
spEllipse::spEllipse(Real major_radius, Real minor_radius)
    : m_major_radius_(major_radius), m_minor_radius_(minor_radius) {}
spEllipse::~spEllipse() = default;

void spEllipse::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_major_radius_ = cfg->GetValue<Real>("MajorRadius", m_major_radius_);
    m_minor_radius_ = cfg->GetValue<Real>("MinorRadius", m_minor_radius_);
}
std::shared_ptr<simpla::data::DataEntry> spEllipse::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("MajorRadius", m_major_radius_);
    res->SetValue<Real>("MinorRadius", m_minor_radius_);
    return res;
}
// bool spEllipse::CheckIntersection(point_type const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return false;
//}
// bool spEllipse::CheckIntersection(box_type const &, Real tolerance) const {
//    UNIMPLEMENTED;
//    return false;
//}
// std::shared_ptr<GeoObject> spEllipse::GetIntersectionion(std::shared_ptr<const GeoObject> const &, Real tolerance)
// const {
//    return nullptr;
//}
}  // namespace geometry{
}  // namespace simpla{