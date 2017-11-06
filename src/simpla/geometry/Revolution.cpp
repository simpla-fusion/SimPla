//
// Created by salmon on 17-10-24.
//

#include "Revolution.h"
#include "Circle.h"

namespace simpla {
namespace geometry {
Revolution::Revolution() = default;
Revolution::Revolution(Revolution const &other) = default;
Revolution::~Revolution() = default;
Revolution::Revolution(Axis const &axis, std::shared_ptr<const GeoObject> const &g, Real angle)
    : Swept(axis), m_basis_obj_(g), m_angle_(angle) {}

void Revolution::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_basis_obj_ = GeoObject::New(cfg->Get("BasisObject"));
    m_angle_ = cfg->GetValue<Real>("Angle", m_angle_);
}
std::shared_ptr<simpla::data::DataNode> Revolution::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("BasisObject", m_basis_obj_->Serialize());
    res->SetValue<Real>("Angle", m_angle_);
    return res;
}

}  // namespace geometry{
}  // namespace simpla{