//
// Created by salmon on 17-10-30.
//

#ifndef SIMPLA_POINT_H
#define SIMPLA_POINT_H
#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Surface;
struct Curve;
struct Point : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Point, GeoObject);

   public:
    int GetDimension() const override { return 0; }

    std::shared_ptr<GeoObject> GetBoundary() const final;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POINT_H
