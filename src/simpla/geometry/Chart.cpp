//
// Created by salmon on 17-5-29.
//
#include "Chart.h"
#include "Box.h"
namespace simpla {
namespace geometry {

constexpr Real Chart::m_id_to_coordinates_shift_[8][3];

Chart::Chart() = default;
Chart::~Chart() { TearDown(); };
Chart::Chart(point_type const &origin, point_type const &grid_width)
    : SPObject(), m_origin_(origin), m_grid_width_(grid_width) {
    Update();
}

std::shared_ptr<data::DataNode> Chart::Serialize() const {
    auto tdb = base_type::Serialize();
    if (tdb != nullptr) {
        tdb->SetValue("Level", GetLevel());
        tdb->SetValue("Origin", GetOrigin());
        tdb->SetValue("GridWidth", GetGridWidth());
    }
    return tdb;
}
void Chart::Deserialize(std::shared_ptr<data::DataNode> const &tdb) {
    ASSERT(tdb != nullptr);
    m_origin_ = tdb->GetValue<point_type>("Origin", m_origin_);
    m_grid_width_ = tdb->GetValue<point_type>("GridWidth", m_grid_width_);
};

void Chart::SetOrigin(point_type const &x) { m_origin_ = x; }
point_type const &Chart::GetOrigin() const { return m_origin_; }
void Chart::SetGridWidth(point_type const &x) { m_grid_width_ = x; }
point_type const &Chart::GetGridWidth() const { return m_grid_width_; }
point_type Chart::GetGridWidth(int level) const {
    point_type res = m_grid_width_;
    if (m_level_ < level) {
        res /= static_cast<Real>(1 << (level - m_level_));
    } else if (m_level_ > level) {
        res *= static_cast<Real>(1 << (m_level_ - level));
    }
    return res;
}

void Chart::SetLevel(int level) {
    m_grid_width_ = GetGridWidth(level);
    m_level_ = level;
};
int Chart::GetLevel() const { return m_level_; }
int Chart::GetNDIMS() const { return 3; }

std::shared_ptr<GeoObject> Chart::GetBoundingShape(box_type const &uvw) const {
    return Box::New(map(std::get<0>(uvw)), map(std::get<1>(uvw)));
}
std::shared_ptr<GeoObject> Chart::GetBoundingShape(index_box_type const &b) const {
    return GetBoundingShape(GetBoxUVW(b));
};
}  // namespace geometry {
}  // namespace simpla {
