//
// Created by salmon on 17-10-23.
//

#include "CylindricalSurface.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(CylindricalSurface)

void CylindricalSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_radius_ = cfg->GetValue<Real>("Radius", m_radius_);
}
std::shared_ptr<simpla::data::DataNode> CylindricalSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Radius", m_radius_);
    return res;
}
}  // namespace geometry{
}  // namespace simpla{