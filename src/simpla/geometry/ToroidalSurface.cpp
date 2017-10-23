//
// Created by salmon on 17-10-23.
//

#include "ToroidalSurface.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(ToroidalSurface)

void ToroidalSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_major_radius_ = cfg->GetValue<Real>("MajorRadius", m_major_radius_);
    m_minor_radius_ = cfg->GetValue<Real>("MinorRadius", m_minor_radius_);
}
std::shared_ptr<simpla::data::DataNode> ToroidalSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("MajorRadius", m_major_radius_);
    res->SetValue<Real>("MinorRadius", m_minor_radius_);
    return res;
}

}  // namespace geometry{
}  // namespace simpla{