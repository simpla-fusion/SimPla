//
// Created by salmon on 17-10-30.
//

#ifndef SIMPLA_POINT_H
#define SIMPLA_POINT_H
#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Point : public GeoObject {
    SP_GEO_OBJECT_HEAD(GeoObject, Point);
    Point(Axis const &);
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POINT_H
