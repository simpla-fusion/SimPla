//
// Created by salmon on 17-10-22.
//

#include "Circle.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Circle)
Circle::Circle() = default;
Circle::Circle(Circle const &) = default;
Circle::Circle(Axis const &axis, Real radius, Real alpha0, Real alpha1) : Curve(axis), m_radius_(radius) {
    SetParameterRange(std::isnan(alpha0) ? GetMinParameter() : alpha0, std::isnan(alpha1) ? GetMaxParameter() : alpha1);
}

Circle::~Circle() = default;
void Circle::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_radius_ = cfg->GetValue<Real>("Radius", m_radius_);
}
std::shared_ptr<simpla::data::DataNode> Circle::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Radius", m_radius_);
    return res;
}
std::shared_ptr<Circle> Circle::New3(point_type const &o, point_type const &b, vector_type const &axis) {
    vector_type r = b - o;
    vector_type z = axis / std::sqrt(dot(axis, axis));
    vector_type x = b - o - dot(b - o, z) * z;
    Real radius = std::sqrt(dot(x, x));
    x /= radius;
    vector_type y = cross(z, x);

    return std::shared_ptr<Circle>(new Circle(Axis{o, x, y, z}, radius));
}
bool Circle::TestInside(point_type const &x) const { return 0; }
std::shared_ptr<GeoObject> Circle::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{