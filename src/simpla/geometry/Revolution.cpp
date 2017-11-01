//
// Created by salmon on 17-10-24.
//

#include "Revolution.h"
#include "Circle.h"

namespace simpla {
namespace geometry {
Revolution::Revolution() = default;
Revolution::Revolution(Revolution const &other) = default;
Revolution::Revolution(Axis const &axis, std::shared_ptr<const Surface> const &s) : Swept(axis), m_basis_surface_(s) {}

Revolution::~Revolution() = default;
void Revolution::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Revolution::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("BasisSurface", m_basis_surface_->Serialize());
    return res;
}

/*******************************************************************************************************************/
SP_OBJECT_REGISTER(RevolutionSurface)
RevolutionSurface::RevolutionSurface() = default;
RevolutionSurface::RevolutionSurface(RevolutionSurface const &other) = default;
RevolutionSurface::RevolutionSurface(Axis const &axis, std::shared_ptr<Curve> const &c) : SweptSurface(axis) {}
RevolutionSurface::~RevolutionSurface() = default;
void RevolutionSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> RevolutionSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("BasisCurve", GetBasisCurve()->Serialize());
    return res;
}
bool RevolutionSurface::IsClosed() const { return GetBasisCurve()->IsClosed(); };

}  // namespace geometry{
}  // namespace simpla{