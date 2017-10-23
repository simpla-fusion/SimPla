//
// Created by salmon on 17-10-23.
//

#include "SweptSurface.h"

namespace simpla {
namespace geometry {

void SweptSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_basis_curve_ = std::dynamic_pointer_cast<Curve>(Curve::New(cfg->Get("BasisCurve")));
}
std::shared_ptr<simpla::data::DataNode> SweptSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("BasisCurve", m_basis_curve_->Serialize());
    return res;
}

}  // namespace geometry{
}  // namespace simpla{