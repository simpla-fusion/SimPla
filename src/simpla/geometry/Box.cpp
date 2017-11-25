//
// Created by salmon on 17-10-17.
//

#include "Box.h"
#include <simpla/SIMPLA_config.h>
#include "GeoAlgorithm.h"
#include "GeoObject.h"
#include "gBox.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Box)
Box::Box(Axis const &axis, vector_type const &extents)
    : Solid(axis, gBox::New(), std::make_tuple(point_type{0, 0, 0}, extents)) {}
Box::Box(point_type const &p0, point_type const &p1) : Box(Axis{p0}, p1 - p0) {}
Box::Box(std::initializer_list<std::initializer_list<Real>> const &v)
    : Box(std::make_tuple(point_type(*v.begin()), point_type(*(v.begin() + 1)))) {}
Box::Box(box_type const &b) : Box(std::get<0>(b), std::get<1>(b)) {}
box_type Box::GetBoundingBox() const {
    auto b = GetParameterRange();
    return std::make_tuple(m_axis_.xyz(std::get<0>(b)), m_axis_.xyz(std::get<1>(b)));
};
// point_type Box::xyz(Real u, Real v, Real w) const { return m_axis_.xyz(u, v, w); };
// point_type Box::uvw(Real x, Real y, Real z) const { return m_axis_.uvw(x, y, z); };
bool Box::CheckIntersection(point_type const &b, Real tolerance) const { return CheckPointInBox(b, GetBoundingBox()); };
bool Box::CheckIntersection(box_type const &b, Real tolerance) const {
    return CalBoxOverlappedArea(GetBoundingBox(), b) > SP_EPSILON;
};
bool Box::CheckIntersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    return g->CheckIntersection(GetBoundingBox(), tolerance);
};

}  // namespace geometry
}  // namespace simpla