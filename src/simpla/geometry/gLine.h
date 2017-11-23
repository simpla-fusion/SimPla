//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_GLINE_H
#define SIMPLA_GLINE_H

#include "gCurve.h"
namespace simpla {
namespace geometry {
struct gLine : public gCurve {
    SP_GEO_ENTITY_HEAD(gCurve, gLine, Line)
    explicit gLine(vector_type const& direction) : m_Direction_(direction) {}
    SP_PROPERTY(vector_type, Direction) = {1, 0, 0};
    point_type xyz(Real u) const override { return m_Direction_ * u; };
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GLINE_H
