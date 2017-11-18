//
// Created by salmon on 17-10-23.
//

#include "spCone.h"
#include "ShapeFunction.h"
namespace simpla {
namespace geometry {

void spCone::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    SetAngle(cfg->GetValue("Angle", GetAngle()));
    SetRadius(cfg->GetValue("Radius", GetRadius()));
}
std::shared_ptr<simpla::data::DataEntry> spCone::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Angle", GetAngle());
    res->SetValue("Radius", GetRadius());

    return res;
}
spCone::spCone() = default;
spCone::spCone(spCone const &other) = default;
spCone::spCone(Real angle, Real radius) : Shape(), m_angle_(angle), m_radius_(radius){};
spCone::~spCone() = default;

}  // namespace geometry {
}  // namespace simpla {