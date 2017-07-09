//
// Created by salmon on 17-5-29.
//
#include "Chart.h"
namespace simpla {
namespace model {
struct Chart::pimpl_s {
    point_type m_shift_{0, 0, 0};
    point_type m_rotation_{0, 0, 0};
    point_type m_scale_{1, 1, 1};
};
Chart::Chart(point_type shift, point_type scale, point_type rotate) : SPObject(), m_pimpl_(new pimpl_s) {
    SetShift(shift);
    SetScale(scale);
    SetRotation(rotate);
}
Chart::~Chart(){};

std::shared_ptr<data::DataTable> Chart::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Shift", GetShift());
    p->SetValue("Scale", GetScale());
    p->SetValue("Rotation", GetRotation());

    return p;
}
void Chart::Deserialize(std::shared_ptr<data::DataTable> p) {
    m_pimpl_->m_shift_ = p->GetValue<point_type>("Shift", GetShift());
    m_pimpl_->m_scale_ = p->GetValue<point_type>("Scale", GetScale());
    m_pimpl_->m_rotation_ = p->GetValue<point_type>("Rotation", GetRotation());
};

void Chart::SetShift(point_type const &x) { m_pimpl_->m_shift_ = x; }
point_type const &Chart::GetShift() const { return m_pimpl_->m_shift_; }

void Chart::SetScale(point_type const &x) { m_pimpl_->m_scale_ = x; }
point_type const &Chart::GetScale() const { return m_pimpl_->m_scale_; }

void Chart::SetRotation(point_type const &x) { m_pimpl_->m_rotation_ = x; }
point_type const &Chart::GetRotation() const { return m_pimpl_->m_rotation_; }

point_type Chart::map(point_type const &x) const {
    // FIXME: add rotation
    return x * m_pimpl_->m_scale_ + m_pimpl_->m_shift_;
}
point_type Chart::inv_map(point_type const &x) const {
    // FIXME: add rotation
    return (x - m_pimpl_->m_shift_) / m_pimpl_->m_scale_;
}
}
}