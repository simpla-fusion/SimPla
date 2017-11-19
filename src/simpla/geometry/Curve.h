//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CURVE_H
#define SIMPLA_CURVE_H

#include "GeoEntity.h"
namespace simpla {
namespace geometry {
struct Curve : public GeoEntity {
    SP_GEO_ENTITY_ABS_HEAD(GeoEntity, Curve)
    virtual bool IsClosed() const { return false; }
};

/**
 * the range of parameter is [0,1)
 */
struct ParametricCurve : public Curve {
    SP_GEO_ENTITY_ABS_HEAD(Curve, ParametricCurve)
    virtual point_type xyz(Real u) const = 0;
    //    ParametricCurve() : m_MinU_(-SP_INFINITY), m_MaxU_(SP_INFINITY) {}
    //    SP_PROPERTY(Real, MinU);
    //    SP_PROPERTY(Real, MaxU);
};
struct ParametricCurve2D : public ParametricCurve {
    SP_GEO_ENTITY_ABS_HEAD(ParametricCurve, ParametricCurve2D)
    virtual point2d_type xy(Real u) const = 0;
    point_type xyz(Real u) const override {
        auto p = xy(u);
        return point_type{p[0], p[1], 0};
    };
};
struct Line : public ParametricCurve2D {
    SP_GEO_ENTITY_HEAD(ParametricCurve2D, Line, Line)
    virtual Real x(Real u) const { return (u); };
    point2d_type xy(Real u) const override { return point2d_type{x(u), 0}; };
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CURVE_H
