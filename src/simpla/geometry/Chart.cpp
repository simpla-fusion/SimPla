//
// Created by salmon on 17-5-29.
//
#include "Chart.h"
namespace simpla {
namespace geometry {
struct Chart::pimpl_s {
    point_type m_origin_{0, 0, 0};
    point_type m_scale_{1, 1, 1};
    point_type m_periodic_dimension_ = {0, 0, 0};
};
Chart::Chart(std::string const &s_name) : SPObject(s_name), m_pimpl_(new pimpl_s) {}
Chart::~Chart(){};

std::shared_ptr<data::DataTable> Chart::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Origin", GetOrigin());
    p->SetValue("dx", GetScale());

    return p;
}
void Chart::Deserialize(const std::shared_ptr<data::DataTable> &p) {
    m_pimpl_->m_origin_ = p->GetValue<point_type>("Origin", GetOrigin());
    m_pimpl_->m_scale_ = p->GetValue<point_type>("dx", GetScale());
};

void Chart::SetPeriodicDimension(point_type const &d) { m_pimpl_->m_periodic_dimension_ = d; }
point_type const &Chart::GetPeriodicDimension() const { return m_pimpl_->m_periodic_dimension_; }
void Chart::SetOrigin(point_type const &x) { m_pimpl_->m_origin_ = x; }
void Chart::SetScale(point_type const &x) { m_pimpl_->m_scale_ = x; }

point_type Chart::GetOrigin() const { return m_pimpl_->m_origin_; }
point_type Chart::GetScale() const { return m_pimpl_->m_scale_; }
}
}