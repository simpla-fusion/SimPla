//
// Created by salmon on 17-10-23.
//

#include "LinearExtrusionSurface.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(LinearExtrusionSurface)

void LinearExtrusionSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
}
std::shared_ptr<simpla::data::DataNode> LinearExtrusionSurface::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}
int LinearExtrusionSurface::CheckOverlap(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> LinearExtrusionSurface::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}
}  // namespace geometry{
}  // namespace