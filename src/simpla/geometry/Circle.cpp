//
// Created by salmon on 17-10-22.
//

#include "Circle.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Circle)

void Circle::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_radius_ = cfg->GetValue<Real>("Radius", m_radius_);
}
std::shared_ptr<simpla::data::DataNode> Circle::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Radius", m_radius_);
    return res;
}

}  // namespace geometry{
}  // namespace simpla{