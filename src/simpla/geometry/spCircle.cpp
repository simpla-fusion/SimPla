//
// Created by salmon on 17-10-22.
//

#include "spCircle.h"

namespace simpla {
namespace geometry {
SP_SHAPE_REGISTER(spCircle)
spCircle::spCircle() = default;
spCircle::spCircle(spCircle const &) = default;
spCircle::spCircle(Real radius) : m_radius_(radius) {}

spCircle::~spCircle() = default;
void spCircle::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_radius_ = cfg->GetValue<Real>("Radius", m_radius_);
}
std::shared_ptr<simpla::data::DataEntry> spCircle::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Radius", m_radius_);
    return res;
}
// std::shared_ptr<spCircle> spCircle::New3(point_type const &o, point_type const &b, vector_type const &axis) {
//    vector_type r = b - o;
//    vector_type z = axis / std::sqrt(dot(axis, axis));
//    vector_type x = b - o - dot(b - o, z) * z;
//    Real radius = std::sqrt(dot(x, x));
//    x /= radius;
//    vector_type y = cross(z, x);
//
//    return std::shared_ptr<spCircle>(new spCircle(Axis{o, x, y, z}, radius));
//}

}  // namespace geometry{
}  // namespace simpla{