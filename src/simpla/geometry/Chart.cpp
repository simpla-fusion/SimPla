//
// Created by salmon on 17-5-29.
//
#include "Chart.h"
namespace simpla {
namespace geometry {

Chart::Chart(point_type shift, point_type scale, point_type rotate) : SPObject() {
    SetOrigin(shift);
    SetScale(scale);
    SetRotation(rotate);
}
Chart::~Chart(){};

std::shared_ptr<data::DataTable> Chart::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Origin", GetOrigin());
    p->SetValue("Scale", GetScale());
    p->SetValue("Rotation", GetRotation());

    return p;
}
void Chart::Deserialize(const std::shared_ptr<data::DataTable> &p) {
    m_origin_ = p->GetValue<point_type>("Origin", GetOrigin());
    m_scale_ = p->GetValue<point_type>("Scale", GetScale());
    m_rotation_ = p->GetValue<point_type>("Rotation", GetRotation());
};

void Chart::SetOrigin(point_type const &x) { m_origin_ = x; }
point_type const &Chart::GetOrigin() const { return m_origin_; }

void Chart::SetScale(point_type const &x) { m_scale_ = x; }
point_type const &Chart::GetScale() const { return m_scale_; }

void Chart::SetRotation(point_type const &x) { m_rotation_ = x; }
point_type const &Chart::GetRotation() const { return m_rotation_; }

point_type Chart::GetCellWidth(int level) const { return m_scale_; }
}
}