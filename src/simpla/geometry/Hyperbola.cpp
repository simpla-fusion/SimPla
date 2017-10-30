//
// Created by salmon on 17-10-22.
//

#include "Hyperbola.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Hyperbola)

void Hyperbola::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_major_radius_ = cfg->GetValue<Real>("MajorRadius", m_major_radius_);
    m_minor_radius_ = cfg->GetValue<Real>("MinorRadius", m_minor_radius_);
}
std::shared_ptr<simpla::data::DataNode> Hyperbola::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("MajorRadius", m_major_radius_);
    res->SetValue<Real>("MinorRadius", m_minor_radius_);
    return res;
}
bool Hyperbola::TestIntersection(box_type const &) const {
    UNIMPLEMENTED;
    return false;
}
std::shared_ptr<GeoObject> Hyperbola::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{