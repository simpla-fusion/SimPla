//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_GCIRCLE_H
#define SIMPLA_GCIRCLE_H

#include <simpla/utilities/Constants.h>
#include <cmath>
#include "Surface.h"
#include "gConic.h"
namespace simpla {
namespace geometry {
struct gCircle : public gConic {
    SP_GEO_ENTITY_HEAD(gConic, gCircle, Circle);

    explicit gCircle(Real radius) : gCircle(radius, 0, TWOPI) {}
    explicit gCircle(Real radius, Real angle0, Real angle1)
        : m_Radius_(radius), m_MinAngle_(angle0), m_MaxAngle_(angle1) {}
    bool IsClosed() const override { return GetMaxAngle() - GetMinAngle() >= TWOPI; }

    SP_PROPERTY(Real, Radius)=1.0;
    SP_PROPERTY(Real, MinAngle);
    SP_PROPERTY(Real, MaxAngle);

    point2d_type xy(Real alpha) const override {
        return point2d_type{m_Radius_ * std::cos(alpha), m_Radius_ * std::sin(alpha)};
    };
};

struct gDisk : public Plane {
    SP_GEO_ENTITY_HEAD(Plane, gDisk, Disk);

    explicit gDisk(Real radius) : gDisk(0, radius, 0, TWOPI) {}
    explicit gDisk(Real min_radius, Real max_radius, Real min_angle = 0, Real max_angle = TWOPI)
        : m_MinRadius_(max_radius), m_MaxRadius_(max_radius), m_MinAngle_(min_angle), m_MaxAngle_(max_angle) {}
    SP_PROPERTY(Real, MinRadius);
    SP_PROPERTY(Real, MaxRadius);
    SP_PROPERTY(Real, MinAngle);
    SP_PROPERTY(Real, MaxAngle);

    point2d_type xy(Real r, Real alpha) const override {
        return point2d_type{r * std::cos(alpha), r * std::sin(alpha)};
    };
};
}  //    namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_GCIRCLE_H
