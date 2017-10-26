//
// Created by salmon on 17-10-26.
//

#ifndef SIMPLA_POINTLIST_H
#define SIMPLA_POINTLIST_H

#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct PointList : public GeoObject {
    SP_GEO_OBJECT_HEAD(PointList, GeoObject)
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POINTLIST_H
