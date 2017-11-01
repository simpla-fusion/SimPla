//
// Created by salmon on 17-10-17.
//

#include "Box.h"
#include <simpla/SIMPLA_config.h>
#include "GeoAlgorithm.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

SP_OBJECT_REGISTER(Box)
constexpr Real Box::m_parameter_range_[2][3];
constexpr Real Box::m_value_range_[2][3];
Box::Box() = default;
Box::Box(Box const &) = default;
Box::~Box() = default;
Box::Box(point_type const &p0, point_type const &p1) {
    SetAxis(
        Axis{p0, point_type{p1[0] - p0[0], 0, 0}, point_type{0, p1[1] - p0[1], 0}, point_type{0, 0, p1[2] - p0[2]}});
}
Box::Box(std::initializer_list<std::initializer_list<Real>> const &v)
    : Box(point_type{*v.begin()}, point_type{*(v.begin() + 1)}) {}
Box::Box(box_type const &b) : Box(std::get<0>(b), std::get<1>(b)) {}

std::shared_ptr<data::DataNode> Box::Serialize() const {
    auto cfg = base_type::Serialize();
    return cfg;
};
void Box::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

box_type Box::GetParameterRange() const { return utility::make_box(m_parameter_range_); };
box_type Box::GetValueRange() const {
    return std::make_tuple(m_axis_.xyz(utility::make_point(m_value_range_[0])),
                           m_axis_.xyz(utility::make_point(m_value_range_[1])));
};
point_type Box::xyz(Real u, Real v, Real w) const { return m_axis_.xyz(u, v, w); };
point_type Box::uvw(Real x, Real y, Real z) const { return m_axis_.uvw(x, y, z); };
bool Box::CheckIntersection(box_type const &b, Real tolerance) const { return TestBoxOverlapped(b, GetBoundingBox()); };

}  // namespace geometry
}  // namespace simpla