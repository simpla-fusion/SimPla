//
// Created by salmon on 17-10-31.
//

#include "ParametricBody.h"
#include "GeoAlgorithm.h"

namespace simpla {
namespace geometry {
ParametricBody::ParametricBody() = default;
ParametricBody::ParametricBody(ParametricBody const &other) = default;
ParametricBody::ParametricBody(Axis const &axis) : GeoObject(axis) {}
ParametricBody::~ParametricBody() = default;

std::shared_ptr<data::DataNode> ParametricBody::Serialize() const { return base_type::Serialize(); };
void ParametricBody::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<Surface> ParametricBody::GetBoundarySurface() const {
    return std::dynamic_pointer_cast<Surface>(m_surface_);
}
box_type ParametricBody::GetBoundingBox() const { return GetValueRange(); };

box_type const &ParametricBody::GetParameterRange() const { return m_parameter_range_; }
box_type const &ParametricBody::GetValueRange() const { return m_value_range_; }
void ParametricBody::SetParameterRange(box_type const &b) { m_parameter_range_ = b; }
void ParametricBody::SetValueRange(box_type const &b) { m_value_range_ = b; }
point_type ParametricBody::xyz(point_type const &u) const { return xyz(u[0], u[1], u[2]); }
point_type ParametricBody::uvw(point_type const &x) const { return uvw(x[0], x[1], x[2]); };
std::shared_ptr<Curve> ParametricBody::Intersection(std::shared_ptr<const Curve> const &g, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
std::shared_ptr<Surface> ParametricBody::Intersection(std::shared_ptr<const Surface> const &g, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
std::shared_ptr<Body> ParametricBody::Intersection(std::shared_ptr<const Body> const &g, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
bool ParametricBody::TestIntersection(point_type const &p, Real tolerance) const {
    return TestPointInBox(p, GetBoundingBox());
};
bool ParametricBody::TestIntersection(box_type const &box, Real tolerance) const {
    return TestBoxOverlapped(box, GetBoundingBox());
};

}  // namespace geometry
}  // namespace simpla