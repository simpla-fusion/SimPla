//
// Created by salmon on 17-10-23.
//

#include "ConicalSurface.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(ConicalSurface)

void ConicalSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_radius_ = cfg->GetValue<Real>("Radius", m_radius_);
    m_semi_angle_ = cfg->GetValue<Real>("SemiAngle", m_semi_angle_);
}
std::shared_ptr<simpla::data::DataNode> ConicalSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->GetValue<Real>("Radius", m_radius_);
    res->SetValue<Real>("SemiAngle", m_semi_angle_);
    return res;
}
int ConicalSurface::CheckOverlap(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> ConicalSurface::Intersection(std::shared_ptr<const GeoObject> const &,
                                                        Real tolerance) const {
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{