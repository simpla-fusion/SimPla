//
// Created by salmon on 17-10-18.
//
#include "Body.h"
namespace simpla {
namespace geometry {

std::shared_ptr<data::DataNode> Body::Serialize() const {
    auto cfg = base_type::Serialize();
    return cfg;
};
void Body::Deserialize(std::shared_ptr<data::DataNode> const& cfg) { base_type::Deserialize(cfg); }
// std::shared_ptr<GeoObject> Surface::GetBoundary() const {
//    UNIMPLEMENTED;
//    return nullptr;
//}

}  // namespace geometry
}  // namespace simpla