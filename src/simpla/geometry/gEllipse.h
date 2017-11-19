//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_GELLIPSE_H
#define SIMPLA_GELLIPSE_H

#include <simpla/utilities/Constants.h>
#include "Surface.h"
#include "gConic.h"
namespace simpla {
namespace geometry {
struct gEllipse : public gConic {
    SP_GEO_ENTITY_HEAD(GeoEntity, gEllipse, Ellipse);

    explicit gEllipse(Real major_radius, Real minor_radius, Real angle = TWOPI)
        : gEllipse(major_radius, minor_radius, 0, angle) {}

    explicit gEllipse(Real major_radius, Real minor_radius, Real angle0, Real angle1)
        : m_MajorRadius_(major_radius), m_MinorRadius_(minor_radius) {}

    bool IsClosed() const override { return GetMaxAngle() - GetMinAngle() >= TWOPI; }

    SP_PROPERTY(Real, MajorRadius);
    SP_PROPERTY(Real, MinorRadius);
    SP_PROPERTY(Real, MinAngle);
    SP_PROPERTY(Real, MaxAngle);
    point2d_type xy(Real alpha) const override {
        return point2d_type{m_MajorRadius_ * std::cos((alpha)), m_MinorRadius_ * std::sin((alpha))};
    };
};

struct gEllipseDisk : public Plane {
    SP_GEO_ENTITY_HEAD(Plane, gEllipseDisk, EllipseDisk);

    explicit gEllipseDisk(Real major_radius, Real minor_radius) : gEllipseDisk(major_radius, minor_radius, 0, TWOPI) {}
    explicit gEllipseDisk(Real major_radius, Real minor_radius, Real min_angle, Real max_angle)
        : m_MinorRadius_(minor_radius), m_MajorRadius_(major_radius), m_MinAngle_(min_angle), m_MaxAngle_(max_angle) {}
    SP_PROPERTY(Real, MajorRadius);
    SP_PROPERTY(Real, MinorRadius);
    SP_PROPERTY(Real, MinAngle);
    SP_PROPERTY(Real, MaxAngle);

    point2d_type xy(Real r, Real alpha) const override {
        return point2d_type{r * std::cos(alpha), r * m_MinorRadius_ / m_MajorRadius_ * std::sin(alpha)};
    };
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_GELLIPSE_H
