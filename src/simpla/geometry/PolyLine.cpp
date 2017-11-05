//
// Created by salmon on 17-10-23.
//

#include "PolyLine.h"

namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(PolyLine)
struct PolyLine::pimpl_s {
    bool m_is_closed_ = false;
    bool m_is_periodic_ = false;
    Real m_min_ = 0;
    Real m_max_ = 1;
    std::vector<nTuple<Real, 3>> m_uvw_;
};
PolyLine::PolyLine() : m_pimpl_(new pimpl_s) {}
PolyLine::PolyLine(PolyLine const &other) : BoundedCurve(other), m_pimpl_(new pimpl_s) {}
PolyLine::~PolyLine() { delete m_pimpl_; }
PolyLine::PolyLine(Axis const &axis) : BoundedCurve(axis), m_pimpl_(new pimpl_s){};

bool PolyLine::IsClosed() const { return m_pimpl_->m_is_closed_; }

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

point_type PolyLine::xyz(Real u) const {
    if (m_pimpl_->m_is_periodic_ || m_pimpl_->m_is_closed_) {
        u -= static_cast<int>((u - m_pimpl_->m_min_) / (m_pimpl_->m_max_ - m_pimpl_->m_min_)) *
             (m_pimpl_->m_max_ - m_pimpl_->m_min_);
    }
    //    auto it = m_pimpl_->m_c_list_.rbegin();
    //    while (u < it->first && it != m_pimpl_->m_c_list_.rend()) { --it; };
    //    ASSERT(it != m_pimpl_->m_c_list_.rend());
    //    return it->second->xyz(u);
    return point_type{0, 0, 0};
}
}  // namespace geometry{
}  // namespace simpla{