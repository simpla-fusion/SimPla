//
// Created by salmon on 17-10-24.
//

#include "SweptBody.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(SweptBody)

SweptBody::SweptBody() = default;
SweptBody::SweptBody(SweptBody const &other) = default;
SweptBody::SweptBody(std::shared_ptr<const Surface> const &s, std::shared_ptr<const Curve> const &c)
    : Body(s->GetAxis()),
      m_basis_surface_(s),
      m_shift_curve_(std::dynamic_pointer_cast<const Curve>(c->Moved(point_type{0, 0, 0}))) {}

SweptBody::~SweptBody() = default;
void SweptBody::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_basis_surface_ = std::dynamic_pointer_cast<Surface>(Surface::New(cfg->Get("BasisSurface")));
    m_shift_curve_ = std::dynamic_pointer_cast<Curve>(Curve::New(cfg->Get("ShiftCurve")));
}
std::shared_ptr<simpla::data::DataNode> SweptBody::Serialize() const {
    auto res = base_type::Serialize();
    if (m_basis_surface_ != nullptr) { res->Set("BasisSurface", m_basis_surface_->Serialize()); }
    if (m_shift_curve_ != nullptr) { res->Set("ShiftCurve", m_shift_curve_->Serialize()); }
    return res;
}
std::shared_ptr<const Surface> SweptBody::GetBasisSurface() const { return m_basis_surface_; }
void SweptBody::SetBasisSurface(std::shared_ptr<const Surface> const &c) { m_basis_surface_ = c; }
std::shared_ptr<const Curve> SweptBody::GetShiftCurve() const { return m_shift_curve_; }
void SweptBody::SetShiftCurve(std::shared_ptr<const Curve> const &c) { m_shift_curve_ = c; }

std::tuple<bool, bool, bool> SweptBody::IsClosed() const {
    auto s = GetBasisSurface()->IsClosed();
    auto c = GetShiftCurve()->IsClosed();

    return std::make_tuple(std::get<0>(s), std::get<1>(s), c);
};
std::tuple<bool, bool, bool> SweptBody::IsPeriodic() const {
    auto s = GetBasisSurface()->IsPeriodic();
    auto c = GetShiftCurve()->IsPeriodic();
    return std::make_tuple(std::get<0>(s), std::get<1>(s), c);
};
nTuple<Real, 3> SweptBody::GetPeriod() const {
    auto s = GetBasisSurface()->GetPeriod();
    auto c = GetShiftCurve()->GetPeriod();
    return nTuple<Real, 3>{s[0], s[1], c};
};
nTuple<Real, 3> SweptBody::GetMinParameter() const {
    auto s = GetBasisSurface()->GetMinParameter();
    auto c = GetShiftCurve()->GetMinParameter();
    return nTuple<Real, 3>{s[0], s[1], c};
}
nTuple<Real, 3> SweptBody::GetMaxParameter() const {
    auto s = GetBasisSurface()->GetMaxParameter();
    auto c = GetShiftCurve()->GetMaxParameter();
    return nTuple<Real, 3>{s[0], s[1], c};
}
point_type SweptBody::Value(Real u, Real v, Real w) const {
    return GetBasisSurface()->Value(u, v) + m_shift_curve_->Value(w);
};
int SweptBody::CheckOverlap(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> SweptBody::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{