//
// Created by salmon on 17-10-20.
//

#include "Plane.h"
#include "GeoAlgorithm.h"
#include "Line.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Plane)
Plane::Plane() = default;
Plane::Plane(Plane const &) = default;
Plane::Plane(Axis const &axis) : Surface(axis) {
    SetParameterRange(std::make_tuple(GetMinParameter(), GetMaxParameter()));
}
Plane::Plane(point_type const &o, vector_type const &x, vector_type const &y) : Plane(Axis(o, x, y)) {}
Plane::~Plane() = default;
void Plane::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Plane::Serialize() const { return base_type::Serialize(); }
point_type Plane::Value(Real u, Real v) const { return m_axis_.uvw(u, v, 0); }

std::shared_ptr<GeoObject> Plane::GetBoundary() const { return base_type::GetBoundary(); }
box_type Plane::GetBoundingBox() const { return base_type::GetBoundingBox(); };
bool Plane::TestIntersection(box_type const &) const { return false; }
bool Plane::TestInside(point_type const &p, Real tolerance) const {
    return TestPointOnPlane(p, m_axis_.o, m_axis_.z, tolerance);
}
bool Plane::TestInsideUV(Real u, Real v, Real tolerance) const { return true; }
std::shared_ptr<GeoObject> Plane::Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    return nullptr;
}
point_type Plane::Value(point_type const &uvw) const { return m_axis_.xyz(uvw); }

}  // namespace geometry{
}  // namespace simpla{impla