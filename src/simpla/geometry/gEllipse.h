//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_GELLIPSE_H
#define SIMPLA_GELLIPSE_H

#include <simpla/utilities/Constants.h>
#include "gCone.h"
#include "gPlane.h"
#include "gSurface.h"
namespace simpla {
namespace geometry {
struct gEllipse : public gConic {
    SP_GEO_ENTITY_HEAD(GeoEntity, gEllipse, Ellipse);

    explicit gEllipse(Real major_radius, Real minor_radius)
        : m_MajorRadius_(major_radius), m_MinorRadius_(minor_radius) {}

    bool IsClosed() const override { return true; }

    SP_PROPERTY(Real, MajorRadius) = 1.0;
    SP_PROPERTY(Real, MinorRadius) = 0.5;
    point2d_type xy(Real alpha) const override {
        return point2d_type{m_MajorRadius_ * std::cos((alpha)), m_MinorRadius_ * std::sin((alpha))};
    };
};

struct gEllipseDisk : public gPlane {
    SP_GEO_ENTITY_HEAD(gPlane, gEllipseDisk, EllipseDisk);

    explicit gEllipseDisk(Real major_radius, Real minor_radius)
        : m_MinorRadius_(minor_radius), m_MajorRadius_(major_radius) {}
    SP_PROPERTY(Real, MajorRadius);
    SP_PROPERTY(Real, MinorRadius);

    point2d_type xy(Real r, Real alpha) const override {
        return point2d_type{r * std::cos(alpha), r * m_MinorRadius_ / m_MajorRadius_ * std::sin(alpha)};
    };
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_GELLIPSE_H
