//
// Created by salmon on 17-11-22.
//

#include "gSweeping.h"
#include "gCircle.h"
#include "gLine.h"

namespace simpla {
namespace geometry {
gSweeping::gSweeping(std::shared_ptr<const GeoEntity> const& basis, std::shared_ptr<const gCurve> const& curve,
                     Axis const& r_axis)
    : m_basis_(basis), m_path_(curve), m_r_axis_(r_axis) {}
gSweeping::gSweeping(gSweeping const& other)
    : m_basis_(other.m_basis_), m_path_(other.m_path_), m_r_axis_(other.m_r_axis_) {}
void gSweeping::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const& cfg) {
    base_type::Deserialize(cfg);
    m_r_axis_.Deserialize(cfg->Get("RelativeAxis"));
    SetBasis(GeoEntity::Create(cfg->Get("Basis")));
    SetPath(gCurve::Create(cfg->Get("Path")));
};
std::shared_ptr<simpla::data::DataEntry> gSweeping::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("RelativeAxis", GetRelativeAxis().Serialize());
    res->Set("Basis", GetBasis()->Serialize());
    res->Set("Path", GetPath()->Serialize());
    return res;
};
void gSweeping::SetRelativeAxis(Axis const& c) { m_r_axis_ = c; };
Axis const& gSweeping::GetRelativeAxis() const { return m_r_axis_; };
void gSweeping::SetPath(std::shared_ptr<const gCurve> const& c) { m_path_ = c; };
std::shared_ptr<const gCurve> gSweeping::GetPath() const { return m_path_; };
void gSweeping::SetBasis(std::shared_ptr<const GeoEntity> const& b) { m_basis_ = b; };
std::shared_ptr<const GeoEntity> gSweeping::GetBasis() const { return m_basis_; };

point_type gSweeping::xyz(Real u, Real v, Real w) const {
    return m_r_axis_.xyz(m_basis_->xyz(u, v, 0)) + m_path_->xyz(w);
};

}  // namespace geometry {
}  // namespace simpla {