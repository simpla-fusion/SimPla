//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_GCONIC_H
#define SIMPLA_GCONIC_H
#include "gCurve.h"
#include "gSurface.h"
namespace simpla {
namespace geometry {
struct gConic : public gCurve2D {
    SP_GEO_ENTITY_ABS_HEAD(gCurve2D, gConic)
};
struct gConicSurface : public gSurface {
    SP_GEO_ENTITY_ABS_HEAD(gSurface, gConicSurface)
};
}  //  namespace geometry{
}  // namespace simpla
#endif  // SIMPLA_GCONIC_H
