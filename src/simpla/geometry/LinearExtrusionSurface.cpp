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
}  // namespace geometry{
}  // namespace