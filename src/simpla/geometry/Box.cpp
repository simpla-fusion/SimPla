//
// Created by salmon on 17-10-17.
//

#include "Box.h"
#include <simpla/SIMPLA_config.h>
#include "GeoAlgorithm.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Box)
Box::Box() = default;
Box::Box(Box const &) = default;
Box::~Box() = default;
Box::Box(point_type const &p0, point_type const &p1) {
    SetAxis(
        Axis{p0, point_type{p1[0] - p0[0], 0, 0}, point_type{0, p1[1] - p0[1], 0}, point_type{0, 0, p1[2] - p0[2]}});
}
Box::Box(std::initializer_list<std::initializer_list<Real>> const &v)
    : Box(point_type(*v.begin()), point_type(*(v.begin() + 1))) {}
Box::Box(box_type const &b) : Box(std::get<0>(b), std::get<1>(b)) {}
Box::Box(vector_type const &extents) : m_extents_(extents) {}

std::shared_ptr<data::DataEntry> Box::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Extents", m_extents_);
    return res;
};
void Box::Deserialize(std::shared_ptr<data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_extents_ = cfg->GetValue("Extents", m_extents_);
}
box_type Box::GetBoundingBox() const { return std::make_tuple(m_axis_.o, m_axis_.xyz(m_extents_)); };
point_type Box::xyz(Real u, Real v, Real w) const { return m_axis_.xyz(u, v, w); };
point_type Box::uvw(Real x, Real y, Real z) const { return m_axis_.uvw(x, y, z); };
bool Box::CheckIntersection(box_type const &b, Real tolerance) const {
    return CheckBoxOverlapped(b, GetBoundingBox());
};
bool Box::CheckIntersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    bool res = false;
    if (auto b = std::dynamic_pointer_cast<const Box>(g)) {
        res = CheckBoxOverlapped(g->GetBoundingBox(), GetBoundingBox());
    } else {
        res = base_type::CheckIntersection(g, tolerance);
    }
    return res;
};

}  // namespace geometry
}  // namespace simpla