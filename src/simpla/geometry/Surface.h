//
// Created by salmon on 17-10-19.
//

#ifndef SIMPLA_SURFACE_H
#define SIMPLA_SURFACE_H

#include <simpla/algebra/nTuple.h>
#include <memory>
#include <utility>
#include "GeoEntity.h"

namespace simpla {
namespace geometry {

/**
 * a surface is a generalization of a plane which needs not be flat, that is, the curvature is not necessarily zero.
 */
struct Surface : public GeoEntity {
    SP_GEO_ENTITY_ABS_HEAD(GeoEntity, Surface);
    virtual bool IsClosed() const { return false; }
};

struct ParametricSurface : public Surface {
    SP_GEO_ENTITY_ABS_HEAD(Surface, ParametricSurface)
    //    ParametricSurface() : m_MinU_(-SP_INFINITY), m_MaxU_(SP_INFINITY), m_MinV_(-SP_INFINITY), m_MaxV_(SP_INFINITY)
    //    {}
    //
    //    SP_PROPERTY(Real, MinU);
    //    SP_PROPERTY(Real, MaxU);
    //    SP_PROPERTY(Real, MinV);
    //    SP_PROPERTY(Real, MaxV);
    virtual point_type xyz(Real u, Real v) const = 0;
};
struct ParametricSurface2D : public ParametricSurface {
    SP_GEO_ENTITY_ABS_HEAD(ParametricSurface, ParametricSurface2D)

    virtual point2d_type xy(Real u, Real v) const = 0;
    point_type xyz(Real u, Real v) const override {
        auto p = xy(u, v);
        return point_type{p[0], p[1], 0};
    };
};
struct Plane : public ParametricSurface2D {
    SP_GEO_ENTITY_ABS_HEAD(ParametricSurface2D, Plane)
    point2d_type xy(Real u, Real v) const override { return point2d_type{(u), (v)}; };
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SURFACE_H
