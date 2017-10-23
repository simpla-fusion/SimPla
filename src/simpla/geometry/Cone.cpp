//
// Created by salmon on 17-10-23.
//

#include "Cone.h"
namespace simpla {
namespace geometry {

std::shared_ptr<simpla::data::DataNode> Cone::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("SemiAngle", m_semi_angle_);
    return res;
};
void Cone::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    m_semi_angle_=cfg->GetValue("SemiAngle", m_semi_angle_);
}
}  // namespace geometry {
}  // namespace simpla {