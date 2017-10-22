//
// Created by salmon on 17-10-21.
//
#include "Line.h"
namespace simpla {
namespace geometry {

std::shared_ptr<data::DataNode> Line::Serialize() const {
    auto cfg = base_type::Serialize();
    cfg->SetValue("Vertices", m_p);
    return cfg;
};
void Line::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    m_p = cfg->GetValue("Vertices", m_p);
}
// box_type Line::GetBoundingBox() const { return box_type{m_p[0], m_p[1]}; };
// bool Line::CheckInside(point_type const& x, Real tolerance) const { return false; };
// std::shared_ptr<GeoObject> Line::GetBoundary() const { return nullptr; };

}  // namespace geometry
}  // namespace simpla