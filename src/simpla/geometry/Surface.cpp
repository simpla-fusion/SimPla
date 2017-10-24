//
// Created by salmon on 17-10-19.
//

#include "Surface.h"
#include <simpla/SIMPLA_config.h>
#include "Curve.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

std::shared_ptr<data::DataNode> Surface::Serialize() const { return base_type::Serialize(); };
void Surface::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

int Surface::FindIntersection(std::shared_ptr<const Curve> const &, std::vector<Real> &, Real tolerance) const {
    UNIMPLEMENTED;
    return 0;
}

// std::shared_ptr<GeoObject> Surface::GetBoundary() const {
//    UNIMPLEMENTED;
//    return nullptr;
//}

}  // namespace geometry
}  // namespace simpla