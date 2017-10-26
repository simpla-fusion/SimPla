//
// Created by salmon on 17-10-23.
//

#include "RevolutionSurface.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(RevolutionSurface)

void RevolutionSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> RevolutionSurface::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}
int RevolutionSurface::CheckOverlap(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> RevolutionSurface::Intersection(std::shared_ptr<const GeoObject> const &,
                                                           Real tolerance) const {
    return nullptr;
}
}  // namespace geometry{
}  // namespace