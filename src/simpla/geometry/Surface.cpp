//
// Created by salmon on 17-10-19.
//

#include "Surface.h"
#include <simpla/SIMPLA_config.h>
#include "Curve.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {
Surface::Surface() = default;
Surface::~Surface() = default;

std::shared_ptr<data::DataNode> Surface::Serialize() const {
    auto cfg = base_type::Serialize();
    return cfg;
};
void Surface::Deserialize(std::shared_ptr<data::DataNode> const& cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<Curve> Surface::GetBoundary() const {
    UNIMPLEMENTED;
    return nullptr;
}

}  // namespace geometry
}  // namespace simpla