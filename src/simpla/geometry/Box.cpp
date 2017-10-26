//
// Created by salmon on 17-10-17.
//

#include "Box.h"
#include "GeoObject.h"
#include "simpla/SIMPLA_config.h"

namespace simpla {
namespace geometry {

SP_OBJECT_REGISTER(Box)

Box::Box() = default;
Box::Box(Box const &) = default;
Box::~Box() = default;

std::shared_ptr<data::DataNode> Box::Serialize() const {
    auto cfg = base_type::Serialize();
    return cfg;
};
void Box::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

int Box::CheckOverlap(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> Box::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}
}  // namespace geometry
}  // namespace simpla