//
// Created by salmon on 17-10-21.
//
#include "Line.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Line)
std::shared_ptr<data::DataNode> Line::Serialize() const { return base_type::Serialize(); };
void Line::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
bool Line::TestIntersection(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> Line::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}
}  // namespace geometry
}  // namespace simpla