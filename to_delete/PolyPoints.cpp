//
// Created by salmon on 17-10-26.
//

#include "PolyPoints.h"
#include <utility>
namespace simpla {
namespace geometry {
PolyPoints::PolyPoints() = default;
PolyPoints::PolyPoints(PolyPoints const &) = default;
PolyPoints::PolyPoints(Axis const &axis) : GeoObject(axis){};

PolyPoints::~PolyPoints() = default;
std::shared_ptr<simpla::data::DataEntry> PolyPoints::Serialize() const { return base_type::Serialize(); };
void PolyPoints::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) { base_type::Deserialize(cfg); };

}  // namespace geometry
}  // namespace simpla