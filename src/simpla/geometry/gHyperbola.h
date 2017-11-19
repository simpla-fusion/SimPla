//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_HYPERBOLA_H
#define SIMPLA_HYPERBOLA_H

#include "gConic.h"
namespace simpla {
namespace geometry {

struct gHyperbola : public gConic {
    SP_GEO_ENTITY_HEAD(gConic, gHyperbola, Hyperbola)
    gHyperbola(Real major_radius, Real minor_radius) : m_MajorRadius_(major_radius), m_MinorRadius_(minor_radius) {}
    SP_PROPERTY(Real, MajorRadius);
    SP_PROPERTY(Real, MinorRadius);

    point2d_type xy(Real alpha) const override {
        return point2d_type{m_MajorRadius_ * std::cosh((alpha)), m_MinorRadius_ * std::sinh((alpha))};
    };
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_HYPERBOLA_H
