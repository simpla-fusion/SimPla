//
// Created by salmon on 17-10-23.
//

#include "SweptSurface.h"

namespace simpla {
namespace geometry {

void SweptSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_direction_ = cfg->GetValue("Direction", m_direction_);
    m_basis_curve_ = Curve::New(cfg->Get("BasisCurve"));
}
std::shared_ptr<simpla::data::DataNode> SweptSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Direction", m_direction_);
    res->Set("BasisCurve", m_basis_curve_->Serialize());
    return res;
}

}  // namespace geometry{
}  // namespace simpla{