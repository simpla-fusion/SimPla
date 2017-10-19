//
// Created by salmon on 17-10-18.
//
#include "Body.h"
#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"
namespace simpla {
namespace geometry {
Body::Body() = default;
Body::~Body() = default;

std::shared_ptr<data::DataNode> Body::Serialize() const {
    auto cfg = base_type::Serialize();
    return cfg;
};
void Body::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

std::shared_ptr<Surface> Body::GetBoundary() const {
    UNIMPLEMENTED;
    return nullptr;
}
bool Body::CheckInside(point_type const &x) const {
    UNIMPLEMENTED;
    return false;
}

}  // namespace geometry
}  // namespace simpla