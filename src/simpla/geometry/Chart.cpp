//
// Created by salmon on 17-5-29.
//
#include "Chart.h"
#include "Box.h"
#include "Edge.h"
#include "Face.h"
#include "Solid.h"
#include "Sweep.h"
namespace simpla {
namespace geometry {

constexpr Real Chart::m_id_to_coordinates_shift_[8][3];

Chart::Chart() = default;
Chart::Chart(Chart const &) = default;
Chart::~Chart() { TearDown(); };
Chart::Chart(point_type const &origin, point_type const &grid_width) : m_origin_(origin), m_grid_width_(grid_width) {
    Update();
}

std::shared_ptr<data::DataEntry> Chart::Serialize() const {
    auto tdb = base_type::Serialize();
    if (tdb != nullptr) {
        tdb->SetValue("Level", GetLevel());
        tdb->SetValue("Origin", GetOrigin());
        tdb->SetValue("GridWidth", GetGridWidth());
    }
    return tdb;
}
void Chart::Deserialize(std::shared_ptr<data::DataEntry> const &tdb) {
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

std::shared_ptr<Face> Chart::GetCoordinateFace(point_type const &o, int normal, Real u, Real v) const {
    return make_Sweep(GetCoordinateEdge(o, (normal + 1) % 3, u), GetCoordinateEdge(o, (normal + 2) % 3, v));
}
std::shared_ptr<Solid> Chart::GetCoordinateBox(point_type const &o, Real u, Real v, Real w) const {
    return make_Sweep(GetCoordinateFace(o, 2, u, v), GetCoordinateEdge(o, 2, w));
}
std::shared_ptr<Solid> Chart::GetCoordinateBox(box_type const &b) const {
    vector_type l = std::get<1>(b) - std::get<0>(b);
    return make_Sweep(GetCoordinateFace(o, 2, l[0], l[1]), GetCoordinateEdge(o, 2, l[2]));
}
std::shared_ptr<Edge> Chart::GetCoordinateEdge(index_tuple const &x0, int normal, size_type u) const {
    return GetCoordinateEdge(uvw(x0), normal, u * m_grid_width_[normal]);
};
std::shared_ptr<Face> Chart::GetCoordinateFace(index_tuple const &x0, int normal, size_type u, size_type v) const {
    return GetCoordinateFace(uvw(x0), normal, u * m_grid_width_[(normal + 1) % 3], v * m_grid_width_[(normal + 2) % 3]);
};
std::shared_ptr<Solid> Chart::GetCoordinateBox(index_tuple const &b, size_type u, size_type v, size_type w) const {
    return GetCoordinateBox(uvw(b), u * m_grid_width_[0], v * m_grid_width_[1], w * m_grid_width_[2]);
};
std::shared_ptr<Solid> Chart::GetCoordinateBox(index_box_type const &b) const {
    return GetCoordinateBox(GetBoxUVW(b));
};

}  // namespace geometry {
}  // namespace simpla {
