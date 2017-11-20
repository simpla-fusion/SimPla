/**
* @file polygon.h
* @author salmon
* @date 2015-11-17.
*/
#ifndef SIMPLA_GPOLYGON_H
#define SIMPLA_GPOLYGON_H

#include <simpla/SIMPLA_config.h>
#include <vector>
#include "gBoundedCurve.h"
namespace simpla {
namespace geometry {

struct gPolygon2D : public gBoundedCurve2D {
    SP_GEO_ENTITY_HEAD(gBoundedCurve2D, gPolygon2D, Polygon2D)
};

struct gPolygon : public gBoundedCurve {
    SP_GEO_ENTITY_HEAD(gBoundedCurve, gPolygon, Polygon)
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GPOLYGON_H
