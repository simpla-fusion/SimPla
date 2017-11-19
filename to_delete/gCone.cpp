//
// Created by salmon on 17-10-23.
//

#include "gCone.h"
#include "ShapeFunction.h"
namespace simpla {
namespace geometry {

void gCone::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    SetAngle(cfg->GetValue("Angle", GetAngle()));
    SetRadius(cfg->GetValue("Radius", GetRadius()));
}
std::shared_ptr<simpla::data::DataEntry> gCone::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Angle", GetAngle());
    res->SetValue("Radius", GetRadius());

    return res;
}
gCone::gCone() = default;
gCone::spCone(gCone const &other) = default;
gCone::gCone(Real angle, Real radius) : GeoEntity(), m_angle_(angle), m_radius_(radius){};
gCone::~gCone() = default;

}  // namespace geometry {
}  // namespace simpla {