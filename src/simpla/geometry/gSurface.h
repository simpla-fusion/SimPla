//
// Created by salmon on 17-10-19.
//

#ifndef SIMPLA_GSURFACE_H
#define SIMPLA_GSURFACE_H

#include <simpla/algebra/nTuple.h>
#include <memory>
#include <utility>
#include "GeoEntity.h"

namespace simpla {
namespace geometry {

/**
 * a surface is a generalization of a plane which needs not be flat, that is, the curvature is not necessarily zero.
 */
struct gSurface : public GeoEntity {
    SP_GEO_ENTITY_ABS_HEAD(GeoEntity, gSurface);
    virtual bool IsClosed() const { return false; }
    point_type xyz(Real u, Real v, Real w) const override { return xyz(u, v); }
    virtual point_type xyz(Real u, Real v) const = 0;
    point_type xyz(point2d_type const& p) const { return xyz(p[0], p[1]); };
};

// struct gPlane : public ParametricSurface2D {
//    SP_GEO_ENTITY_ABS_HEAD(ParametricSurface2D, gPlane)
//    point2d_type xy(Real u, Real v) const override { return point2d_type{(u), (v)}; };
//};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GSURFACE_H
