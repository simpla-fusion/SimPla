//
// Created by salmon on 17-10-23.
//

#include "RevolutionSurface.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(RevolutionSurface)

void RevolutionSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_origin_ = cfg->GetValue("Origin", m_origin_);
}
std::shared_ptr<simpla::data::DataNode> RevolutionSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Origin", m_origin_);
    return res;
}
}  // namespace geometry{
}  // namespace