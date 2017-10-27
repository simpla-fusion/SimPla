//
// Created by salmon on 17-10-23.
//

#include "Polyline.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Polyline)
struct Polyline::pimpl_s {
    bool m_is_closed_ = false;
    bool m_is_periodic_ = false;
    Real m_min_ = 0;
    Real m_max_ = 1;
    std::vector<nTuple<Real, 3>> m_uvw_;
};
Polyline::Polyline() : m_pimpl_(new pimpl_s) {}
Polyline::Polyline(Polyline const &other) : Curve(other), m_pimpl_(new pimpl_s) {}
Polyline::~Polyline() { delete m_pimpl_; }

bool Polyline::IsClosed() const { return m_pimpl_->m_is_closed_; }
bool Polyline::IsPeriodic() const { return m_pimpl_->m_is_periodic_; }

// void Polyline::SetClosed(bool flag) { m_pimpl_->m_is_closed_ = true; }
// void Polyline::SetPeriod(Real l) {
//    m_pimpl_->m_max_ = m_pimpl_->m_min_ + l;
//    m_pimpl_->m_is_periodic_ = true;
//    m_pimpl_->m_is_closed_ = true;
//}
Real Polyline::GetPeriod() const { return m_pimpl_->m_max_ - m_pimpl_->m_min_; }
Real Polyline::GetMinParameter() const { return m_pimpl_->m_min_; }
Real Polyline::GetMaxParameter() const { return m_pimpl_->m_max_; }

void Polyline::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_pimpl_->m_is_closed_ = cfg->GetValue<bool>("IsClosed", m_pimpl_->m_is_closed_);
    m_pimpl_->m_is_periodic_ = cfg->GetValue<bool>("IsPeriodic", m_pimpl_->m_is_periodic_);
    m_pimpl_->m_min_ = cfg->GetValue<Real>("MinParameter", m_pimpl_->m_min_);
    m_pimpl_->m_max_ = cfg->GetValue<Real>("MaxParameter", m_pimpl_->m_max_);
}
std::shared_ptr<simpla::data::DataNode> Polyline::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<bool>("IsClosed", m_pimpl_->m_is_closed_);
    res->SetValue<bool>("IsPeriodic", m_pimpl_->m_is_periodic_);
    res->SetValue<Real>("MinParameter", m_pimpl_->m_min_);
    res->SetValue<Real>("MaxParameter", m_pimpl_->m_max_);
    return res;
}

point_type Polyline::Value(Real u) const {}

int Polyline::CheckOverlap(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> Polyline::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{