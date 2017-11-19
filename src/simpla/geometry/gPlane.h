//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_GPLANE_H
#define SIMPLA_GPLANE_H
#include "Surface.h"
namespace simpla {
namespace geometry {
struct gPlane : public ParametricSurface2D {
    SP_GEO_ENTITY_HEAD(ParametricSurface2D, gPlane, Plane)
    point2d_type xy(Real u, Real v) const override { return point2d_type{u, v}; };
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GPLANE_H
