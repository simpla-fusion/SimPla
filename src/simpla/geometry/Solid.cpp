//
// Created by salmon on 17-11-18.
//

#include "Solid.h"
namespace simpla {
namespace geometry {
Solid::Solid() = default;
Solid::~Solid() = default;
Solid::Solid(Solid const &other) = default;
Solid::Solid(std::shared_ptr<const Body> const &body, point_type const &u_min, point_type const &u_max)
    : m_body_(body), m_range_{u_min, u_max} {};
void Solid::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_range_ = cfg->GetValue("ParameterRange", m_range_);
};
std::shared_ptr<simpla::data::DataEntry> Solid::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("ParameterRange", m_range_);
    return res;
};
}  // namespace geometry
}  // namespace simpla