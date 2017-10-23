//
// Created by salmon on 17-10-21.
//
#include "Line.h"
namespace simpla {
namespace geometry {
    SP_OBJECT_REGISTER(Line)

std::shared_ptr<data::DataNode> Line::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Origin", m_origin_);
    res->SetValue("XAxis", m_x_axis_);
    return res;
};
void Line::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    m_origin_ = cfg->GetValue("Origin", m_origin_);
    m_x_axis_ = cfg->GetValue("XAxis", m_x_axis_);
}
// box_type Line::GetBoundingBox() const { return box_type{m_p[0], m_p[1]}; };
// bool Line::CheckInside(point_type const& x, Real tolerance) const { return false; };
// std::shared_ptr<GeoObject> Line::GetBoundary() const { return nullptr; };

}  // namespace geometry
}  // namespace simpla