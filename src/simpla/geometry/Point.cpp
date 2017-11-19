//
// Created by salmon on 17-10-30.
//

#include "Point.h"
#include "Box.h"
namespace simpla {
namespace geometry {
Point::Point() = default;
Point::Point(Point const &) = default;
Point::~Point() = default;
Point::Point(Axis const &axis) : GeoObject(axis) {}

}  // namespace geometry
}  // namespace simpla