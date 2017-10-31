//
// Created by salmon on 17-10-22.
//

#include "Parabola.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Parabola)

void Parabola::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_focal_ = cfg->GetValue<Real>("Focal", m_focal_);
}
std::shared_ptr<simpla::data::DataNode> Parabola::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Focal", m_focal_);
    return res;
}
bool Parabola::TestIntersection(box_type const &, Real tolerance) const { return 0; }
std::shared_ptr<GeoObject> Parabola::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{