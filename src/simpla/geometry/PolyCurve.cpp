//
// Created by salmon on 17-10-23.
//

#include "PolyCurve.h"

namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(PolyCurve)
struct PolyCurve::pimpl_s {
    bool m_is_closed_ = false;
    bool m_is_periodic_ = false;
    Real m_min_ = 0;
    Real m_max_ = 1;
    std::list<std::pair<Real, std::shared_ptr<Curve>>> m_c_list_;
};
PolyCurve::PolyCurve() : m_pimpl_(new pimpl_s) {}
PolyCurve::PolyCurve(PolyCurve const &other) : Curve(other), m_pimpl_(new pimpl_s) {}
PolyCurve::~PolyCurve() { delete m_pimpl_; }

bool PolyCurve::IsClosed() const { return m_pimpl_->m_is_closed_; }
void PolyCurve::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_pimpl_->m_is_closed_ = cfg->GetValue<bool>("IsClosed", m_pimpl_->m_is_closed_);
    m_pimpl_->m_is_periodic_ = cfg->GetValue<bool>("IsPeriodic", m_pimpl_->m_is_periodic_);
    m_pimpl_->m_min_ = cfg->GetValue<Real>("MinParameter", m_pimpl_->m_min_);
    m_pimpl_->m_max_ = cfg->GetValue<Real>("MaxParameter", m_pimpl_->m_max_);
}
std::shared_ptr<simpla::data::DataNode> PolyCurve::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<bool>("IsClosed", m_pimpl_->m_is_closed_);
    res->SetValue<bool>("IsPeriodic", m_pimpl_->m_is_periodic_);
    res->SetValue<Real>("MinParameter", m_pimpl_->m_min_);
    res->SetValue<Real>("MaxParameter", m_pimpl_->m_max_);
    return res;
}

point_type PolyCurve::xyz(Real u) const {
    if (m_pimpl_->m_is_periodic_ || m_pimpl_->m_is_closed_) {
        u -= static_cast<int>((u - m_pimpl_->m_min_) / (m_pimpl_->m_max_ - m_pimpl_->m_min_)) *
             (m_pimpl_->m_max_ - m_pimpl_->m_min_);
    }
    auto it = m_pimpl_->m_c_list_.rbegin();
    while (u < it->first && it != m_pimpl_->m_c_list_.rend()) { --it; };
    ASSERT(it != m_pimpl_->m_c_list_.rend());
    return it->second->xyz(u);
}
void PolyCurve::PushBack(std::shared_ptr<Curve> const &c, Real length) {
    //    length = std::isnan(length) ? (c->GetMaxParameter() - c->GetMinParameter()) : length;
    //    m_pimpl_->m_c_list_.push_front(std::make_pair(m_pimpl_->m_max_, c));
    //    m_pimpl_->m_max_ += length;
}
void PolyCurve::PushFront(std::shared_ptr<Curve> const &c, Real length) {
    //    length = std::isnan(length) ? (c->GetMaxParameter() - c->GetMinParameter()) : length;
    //    m_pimpl_->m_min_ -= length;
    //    m_pimpl_->m_c_list_.push_front(std::make_pair(m_pimpl_->m_min_, c));
}

void PolyCurve::Foreach(std::function<void(std::shared_ptr<Curve> const &)> const &fun) {
    for (auto &item : m_pimpl_->m_c_list_) { fun(item.second); }
}
void PolyCurve::Foreach(std::function<void(std::shared_ptr<const Curve> const &)> const &fun) const {
    for (auto &item : m_pimpl_->m_c_list_) { fun(item.second); }
}

}  // namespace geometry{
}  // namespace simpla{