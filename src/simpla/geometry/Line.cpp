//
// Created by salmon on 17-10-21.
//
#include "Line.h"
namespace simpla {
namespace geometry {
    SP_OBJECT_REGISTER(Line)

std::shared_ptr<data::DataNode> Line::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Origin", m_axis_.o);
    res->SetValue("XAxis", m_axis_.x);
    return res;
};
void Line::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    m_axis_.o = cfg->GetValue("Origin", m_axis_.o);
    m_axis_.x = cfg->GetValue("XAxis", m_axis_.x);
}
// box_type Line::GetBoundingBox() const { return box_type{m_p[0], m_p[1]}; };
// bool Line::CheckInside(point_type const& x, Real tolerance) const { return false; };
// std::shared_ptr<GeoObject> Line::GetBoundary() const { return nullptr; };

}  // namespace geometry
}  // namespace simpla