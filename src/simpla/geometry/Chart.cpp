//
// Created by salmon on 17-5-29.
//
#include "Chart.h"
#include "Box.h"
namespace simpla {
namespace geometry {
constexpr Real Chart::m_id_to_coordinates_shift_[8][3];
Chart::Chart() = default;
Chart::~Chart() = default;
std::shared_ptr<data::DataNode> Chart::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Axis", m_axis_.Serialize());
    return res;
}
void Chart::Deserialize(std::shared_ptr<data::DataNode> const &tdb) {
    if (tdb != nullptr) { m_axis_.Deserialize(tdb->Get("Axis")); }
};
void Chart::SetOrigin(point_type const &x) { m_axis_.SetOrigin(x); }
point_type const &Chart::GetOrigin() const { return m_axis_.GetOrigin(); }

point_type Chart::GetCellWidth(int level) const {
    point_type res{std::sqrt(dot(m_axis_.x, m_axis_.x)), std::sqrt(dot(m_axis_.y, m_axis_.y)),
                   std::sqrt(dot(m_axis_.z, m_axis_.z))};
    if (m_level_ < level) {
        res /= static_cast<Real>(1 << (level - m_level_));
    } else if (m_level_ > level) {
        res *= static_cast<Real>(1 << (m_level_ - level));
    }

    return res;
}

void Chart::SetLevel(int level) { m_level_ = level; };
int Chart::GetLevel() const { return m_level_; }
int Chart::GetNDIMS() const { return 3; }

}  // namespace geometry {
}  // namespace simpla {
