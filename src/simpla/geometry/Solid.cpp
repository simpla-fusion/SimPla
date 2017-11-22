//
// Created by salmon on 17-11-18.
//

#include "Solid.h"
#include "gBody.h"
namespace simpla {
namespace geometry {

Solid::Solid(Axis const &axis, std::shared_ptr<const gBody> const &body, Real u_min, Real u_max, Real v_min, Real v_max,
             Real w_min, Real w_max)
    : Solid(axis, body, box_type{{u_min, v_min, w_min}, {u_max, v_max, w_max}}) {}
Solid::Solid(Axis const &axis, std::shared_ptr<const gBody> const &body, point_type const &u_min,
             point_type const &u_max)
    : Solid(axis, body, box_type{u_min, u_max}) {}
Solid::Solid(Axis const &axis, std::shared_ptr<const gBody> const &body, box_type const &b)
    : GeoObject(axis), m_body_(body), m_range_{b} {};
void Solid::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_body_ = GeoEntity::CreateAs<gBody>(cfg->Get("gBody"));
    m_range_ = cfg->GetValue("ParameterRange", m_range_);
};
std::shared_ptr<simpla::data::DataEntry> Solid::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("gBody", m_body_->Serialize());
    res->SetValue("ParameterRange", m_range_);
    return res;
};
}  // namespace geometry
}  // namespace simpla