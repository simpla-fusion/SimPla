//
// Created by salmon on 17-10-24.
//

#include "Revolution.h"
#include "spCircle.h"

namespace simpla {
namespace geometry {
Revolution::Revolution() = default;
Revolution::Revolution(Revolution const &other) = default;
Revolution::~Revolution() = default;
Revolution::Revolution(Axis const &axis, std::shared_ptr<const GeoObject> const &g, Real angle)
    : PrimitiveShape(axis), m_basis_obj_(g), m_angle_(angle) {}
Revolution::Revolution(std::shared_ptr<const GeoObject> const &g, Real angle)
    : PrimitiveShape(), m_basis_obj_(g), m_angle_(angle) {}
void Revolution::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_basis_obj_ = GeoObject::New(cfg->Get("BasisObject"));
    m_angle_ = cfg->GetValue<Real>("MinAngle", m_angle_);
}
std::shared_ptr<simpla::data::DataEntry> Revolution::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("BasisObject", m_basis_obj_->Serialize());
    res->SetValue<Real>("Angle", m_angle_);
    return res;
}
point_type Revolution::xyz(Real u, Real v, Real w) const {
    auto b = std::dynamic_pointer_cast<const PrimitiveShape>(GetBasisObject());
    ASSERT(b != nullptr);
    point_type p = b->xyz(u, v, 0);
    Real sinw = std::sin(w);
    Real cosw = std::cos(w);
    return m_axis_.xyz(p[0] * cosw - p[1] * sinw, p[0] * sinw + p[1] * cosw, p[2]);
};
point_type Revolution::uvw(Real x, Real y, Real z) const {
    point_type p = m_axis_.uvw(x, y, z);
    return point_type{std::hypot(p[0], p[1]), p[2], std::atan2(p[1], p[0])};
};
}  // namespace geometry{
}  // namespace simpla{