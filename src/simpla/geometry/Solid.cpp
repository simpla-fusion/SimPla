//
// Created by salmon on 17-11-18.
//

#include "Solid.h"
#include "gBody.h"
namespace simpla {
namespace geometry {
Solid::Solid(std::shared_ptr<const GeoEntity> const &body)
    : Solid(std::dynamic_pointer_cast<const gBody>(body), Axis{}, box_type{{0, 0, 0}, {1, 1, 1}}) {}
Solid::Solid(std::shared_ptr<const gBody> const &body, Axis const &axis, Real u_min, Real u_max, Real v_min, Real v_max,
             Real w_min, Real w_max)
    : Solid(body, axis, box_type{{u_min, v_min, w_min}, {u_max, v_max, w_max}}) {}
Solid::Solid(std::shared_ptr<const gBody> const &body, Axis const &axis, point_type const &u_min,
             point_type const &u_max)
    : Solid(body, axis, box_type{u_min, u_max}) {}
Solid::Solid(std::shared_ptr<const gBody> const &body, Axis const &axis, box_type const &b)
    : GeoObject(axis), m_body_(body), m_range_{b} {};
void Solid::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_body_ = GeoEntity::CreateAs<gBody>(cfg->Get("gBody"));
    m_range_ = cfg->GetValue("ParameterRange", m_range_);
};
std::shared_ptr<simpla::data::DataEntry> Solid::Serialize() const {
    auto res = base_type::Serialize();
    if (m_body_ != nullptr) res->Set("gBody", m_body_->Serialize());
    res->SetValue("ParameterRange", m_range_);
    return res;
};
}  // namespace geometry
}  // namespace simpla