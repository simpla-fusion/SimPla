//
// Created by salmon on 17-10-23.
//

#include "spCone.h"
#include "ShapeFunction.h"
namespace simpla {
namespace geometry {

void spCone::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    SetSemiAngle(cfg->GetValue("SemiAngle", GetSemiAngle()));
}
std::shared_ptr<simpla::data::DataEntry> spCone::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("SemiAngle", GetSemiAngle());
    return res;
}
spCone::spCone() = default;
spCone::spCone(spCone const &other) = default;
spCone::spCone(Real semi_angle) : Shape(), m_semi_angle_(semi_angle){};
spCone::~spCone() = default;

}  // namespace geometry {
}  // namespace simpla {