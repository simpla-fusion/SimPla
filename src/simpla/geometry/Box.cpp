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

Box::Box(point_type const &p0, point_type const &p1)
    : Solid(Axis{p0, point_type{p1[0] - p0[0], 0, 0}, point_type{0, p1[1] - p0[1], 0}, point_type{0, 0, p1[2] - p0[2]}},
            nullptr, p0, p1) {}
Box::Box(point_type const &p0, Real u, Real v, Real w)
    : Solid(Axis{p0, point_type{u, 0, 0}, point_type{0, v, 0}, point_type{0, 0, w}}, nullptr, 0, 0, 0, u, v, w) {}

Box::Box(std::initializer_list<std::initializer_list<Real>> const &v)
    : Box(point_type(*v.begin()), point_type(*(v.begin() + 1))) {}
Box::Box(box_type const &b) : Box(std::get<0>(b), std::get<1>(b)) {}
Box::Box(Axis const &axis, vector_type const &extents) : Solid(axis, nullptr, point_type{0, 0, 0}, extents) {}

// box_type Box::GetBoundingBox() const { return std::make_tuple(m_axis_.o, m_axis_.xyz(m_extents_)); };
// point_type Box::xyz(Real u, Real v, Real w) const { return m_axis_.xyz(u, v, w); };
// point_type Box::uvw(Real x, Real y, Real z) const { return m_axis_.uvw(x, y, z); };
// bool Box::CheckIntersection(box_type const &b, Real tolerance) const {
//    return CheckBoxOverlapped(b, GetBoundingBox());
//};
// bool Box::CheckIntersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
//    bool res = false;
//    if (auto b = std::dynamic_pointer_cast<const Box>(g)) {
//        res = CheckBoxOverlapped(g->GetBoundingBox(), GetBoundingBox());
//    } else {
//        res = base_type::CheckIntersection(g, tolerance);
//    }
//    return res;
//};

}  // namespace geometry
}  // namespace simpla