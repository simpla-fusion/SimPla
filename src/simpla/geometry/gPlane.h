//
// Created by salmon on 17-11-20.
//

#ifndef SIMPLA_GPLANE_H
#define SIMPLA_GPLANE_H

#include "gSurface.h"
namespace simpla {
namespace geometry {
struct gPlane : public gSurface {
    SP_GEO_ENTITY_HEAD(gSurface, gPlane, Plane);
    virtual point2d_type xy(Real u, Real v) const { return point2d_type{u, v}; };
    point_type xyz(Real u, Real v) const override { return point_type{u, v, 0}; };
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GPLANE_H
