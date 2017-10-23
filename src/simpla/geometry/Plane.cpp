//
// Created by salmon on 17-10-20.
//

#include "Plane.h"
namespace simpla {
namespace geometry {
    SP_OBJECT_REGISTER(Plane)

void Plane::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_axis_.o = cfg->GetValue("Origin", m_axis_.o);
    m_axis_.x = cfg->GetValue("XAxis", m_axis_.x);
    m_axis_.y = cfg->GetValue("YAxis", m_axis_.y);
}
std::shared_ptr<simpla::data::DataNode> Plane::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Origin", m_axis_.o);
    res->SetValue("XAxis", m_axis_.x);
    res->SetValue("YAxis", m_axis_.y);
    return res;
}

}  // namespace geometry{
}  // namespace simpla{impla