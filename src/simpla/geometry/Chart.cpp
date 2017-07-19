//
// Created by salmon on 17-5-29.
//
#include "Chart.h"
namespace simpla {
namespace geometry {

Chart::Chart(point_type shift, point_type scale, point_type rotate) {
    SetOrigin(shift);
    SetScale(scale);
    SetRotation(rotate);
}

std::shared_ptr<data::DataTable> Chart::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Level", GetLevel());
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

point_type Chart::GetCellWidth(size_type level) const {
    point_type res = m_scale_;
    if (m_level_ < level) {
        res /= static_cast<Real>(1 << (level - m_level_));
    } else if (m_level_ > level) {
        res *= static_cast<Real>(1 << (m_level_ - level));
    }

    return res;
}

void Chart::SetLevel(size_type level) {
    m_scale_ = GetCellWidth(level);
    m_level_ = level;
};
size_type Chart::GetLevel() const { return m_level_; }
}
}