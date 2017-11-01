//
// Created by salmon on 17-10-30.
//

#include "Point.h"
#include "Box.h"
namespace simpla {
namespace geometry {
Point::Point() = default;
Point::Point(Point const &other) = default;
Point::~Point() = default;
Point::Point(Axis const &axis) : GeoObject(axis) {}
std::shared_ptr<data::DataNode> Point::Serialize() const { return base_type::Serialize(); };
void Point::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<GeoObject> Point::GetBoundary() const { return Box::New(GetBoundingBox()); };

}  // namespace geometry
}  // namespace simpla