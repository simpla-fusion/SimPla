//
// Created by salmon on 17-10-30.
//

#ifndef SIMPLA_POINT_H
#define SIMPLA_POINT_H

#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Point : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Point, GeoObject);

   protected:
    Point() = default;
    Point(Point const &other) = default;
    explicit Point(Axis const &axis) : GeoObject(axis) {}

   public:
    ~Point() override = default;
    point_type Value() const { return m_axis_.o; }
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POINT_H
