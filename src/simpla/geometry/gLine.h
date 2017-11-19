//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_GLINE_H
#define SIMPLA_GLINE_H

#include "Curve.h"
namespace simpla {
namespace geometry {
struct gLine : public ParametricCurve2D {
    SP_GEO_ENTITY_HEAD(ParametricCurve2D, gLine, Line)
    virtual Real x(Real u) const { return (u); };
    point2d_type xy(Real u) const override { return point2d_type{x(u), 0}; };
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GLINE_H
