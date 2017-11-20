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

point_type Chart::GetGridWidth(int level) const {
    point_type res = m_grid_width_;
    if (m_level_ < level) {
        res /= static_cast<Real>(1 << (level - m_level_));
    } else if (m_level_ > level) {
        res *= static_cast<Real>(1 << (m_level_ - level));
    }
    return res;
}
void Chart::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_axis_.Deserialize(cfg->Get("Axis"));
};
std::shared_ptr<simpla::data::DataEntry> Chart::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Axis", m_axis_.Serialize());
    return res;
};

int Chart::GetNDIMS() const { return 3; }
bool Chart::IsValid() const { return m_is_valid_; }
void Chart::Update() {
    m_is_valid_ = true;
    m_grid_width_ = GetGridWidth(GetLevel());
};
void Chart::TearDown() { m_is_valid_ = false; };

Axis Chart::GetLocalAxis(point_type const &o) const {
    auto axis = m_axis_;
    axis.SetOrigin(o);
    return std::move(axis);
}

std::shared_ptr<Face> Chart::GetCoordinateFace(point_type const &o, int normal, Real u, Real v) const {
    return std::dynamic_pointer_cast<Face>(
        MakeSweep(GetCoordinateEdge(o, (normal + 1) % 3, u), GetCoordinateEdge(o, (normal + 2) % 3, v)));
}
std::shared_ptr<Solid> Chart::GetCoordinateBox(point_type const &o, Real u, Real v, Real w) const {
    return std::dynamic_pointer_cast<Solid>(MakeSweep(GetCoordinateFace(o, 2, u, v), GetCoordinateEdge(o, 2, w)));
}
std::shared_ptr<Solid> Chart::GetCoordinateBox(box_type const &b) const {
    vector_type l = std::get<1>(b) - std::get<0>(b);
    return std::dynamic_pointer_cast<Solid>(
        MakeSweep(GetCoordinateFace(std::get<0>(b), 2, l[0], l[1]), GetCoordinateEdge(std::get<0>(b), 2, l[2])));
}
std::shared_ptr<Edge> Chart::GetCoordinateEdge(index_tuple const &x0, int normal, index_type u) const {
    return GetCoordinateEdge(uvw(x0), normal, u * m_grid_width_[normal]);
};
std::shared_ptr<Face> Chart::GetCoordinateFace(index_tuple const &x0, int normal, index_type u, index_type v) const {
    return GetCoordinateFace(uvw(x0), normal, u * m_grid_width_[(normal + 1) % 3], v * m_grid_width_[(normal + 2) % 3]);
};
std::shared_ptr<Solid> Chart::GetCoordinateBox(index_tuple const &b, index_type u, index_type v, index_type w) const {
    return GetCoordinateBox(uvw(b), u * m_grid_width_[0], v * m_grid_width_[1], w * m_grid_width_[2]);
};
std::shared_ptr<Solid> Chart::GetCoordinateBox(index_box_type const &b) const {
    return GetCoordinateBox(GetBoxUVW(b));
};

}  // namespace geometry {
}  // namespace simpla {
