//
// Created by salmon on 17-10-23.
//

#include "SphericalSurface.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(SphericalSurface)

void SphericalSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_radius_ = cfg->GetValue<Real>("Radius", m_radius_);
}
std::shared_ptr<simpla::data::DataNode> SphericalSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Radius", m_radius_);
    return res;
}
}  // namespace geometry{
}  // namespace simpla{