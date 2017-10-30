//
// Created by salmon on 17-10-23.
//

#include "PolyLine.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(PolyLine)
struct PolyLine::pimpl_s {
    bool m_is_closed_ = false;
    bool m_is_periodic_ = false;
    Real m_min_ = 0;
    Real m_max_ = 1;
    std::vector<nTuple<Real, 3>> m_uvw_;
};
PolyLine::PolyLine() : m_pimpl_(new pimpl_s) {}
PolyLine::PolyLine(PolyLine const &other) : Curve(other), m_pimpl_(new pimpl_s) {}
PolyLine::~PolyLine() { delete m_pimpl_; }

bool PolyLine::IsClosed() const { return m_pimpl_->m_is_closed_; }
bool PolyLine::IsPeriodic() const { return m_pimpl_->m_is_periodic_; }

// void Polyline::SetClosed(bool flag) { m_pimpl_->m_is_closed_ = true; }
// void Polyline::SetPeriod(Real l) {
//    m_pimpl_->m_max_ = m_pimpl_->m_min_ + l;
//    m_pimpl_->m_is_periodic_ = true;
//    m_pimpl_->m_is_closed_ = true;
//}
Real PolyLine::GetPeriod() const { return m_pimpl_->m_max_ - m_pimpl_->m_min_; }
Real PolyLine::GetMinParameter() const { return m_pimpl_->m_min_; }
Real PolyLine::GetMaxParameter() const { return m_pimpl_->m_max_; }

void PolyLine::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_pimpl_->m_is_closed_ = cfg->GetValue<bool>("IsClosed", m_pimpl_->m_is_closed_);
    m_pimpl_->m_is_periodic_ = cfg->GetValue<bool>("IsPeriodic", m_pimpl_->m_is_periodic_);
    m_pimpl_->m_min_ = cfg->GetValue<Real>("MinParameter", m_pimpl_->m_min_);
    m_pimpl_->m_max_ = cfg->GetValue<Real>("MaxParameter", m_pimpl_->m_max_);
}
std::shared_ptr<simpla::data::DataNode> PolyLine::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<bool>("IsClosed", m_pimpl_->m_is_closed_);
    res->SetValue<bool>("IsPeriodic", m_pimpl_->m_is_periodic_);
    res->SetValue<Real>("MinParameter", m_pimpl_->m_min_);
    res->SetValue<Real>("MaxParameter", m_pimpl_->m_max_);
    return res;
}

point_type PolyLine::Value(Real u) const { return point_type{0, 0, 0}; }

bool PolyLine::TestIntersection(box_type const &) const { return false; }
std::shared_ptr<GeoObject> PolyLine::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{