//
// Created by salmon on 17-10-30.
//

#include "Point.h"
#include "Box.h"
namespace simpla {
namespace geometry {

Point::Point(Axis const &axis) : GeoObject(axis) {}
std::shared_ptr<data::DataEntry> Point::Serialize() const { return base_type::Serialize(); };
void Point::Deserialize(std::shared_ptr<const data::DataEntry> const &cfg) { base_type::Deserialize(cfg); }

}  // namespace geometry
}  // namespace simpla