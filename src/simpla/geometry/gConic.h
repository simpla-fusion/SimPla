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
    template <typename... Args>
    explicit gConic(Args&&... args) : gCurve2D(std::forward<Args>(args)...) {}
};
struct gConicSurface : public gSurface {
    SP_GEO_ENTITY_ABS_HEAD(gSurface, gConicSurface)
    template <typename... Args>
    explicit gConicSurface(Args&&... args) : gSurface(std::forward<Args>(args)...) {}
};
}  //  namespace geometry{
}  // namespace simpla
#endif  // SIMPLA_GCONIC_H
