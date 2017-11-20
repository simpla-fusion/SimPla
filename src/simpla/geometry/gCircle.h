//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_GCIRCLE_H
#define SIMPLA_GCIRCLE_H

#include <simpla/data/Configurable.h>
#include <simpla/utilities/Constants.h>
#include <cmath>
#include "GeoEntity.h"
#include "gConic.h"
#include "gPlane.h"
#include "gSurface.h"
namespace simpla {
namespace geometry {
struct gCircle : public gConic {
    SP_GEO_ENTITY_HEAD(gConic, gCircle, Circle);

    explicit gCircle(Real radius) : gCircle(radius, 0, TWOPI) {}
    explicit gCircle(Real radius, Real angle0, Real angle1)
        : m_Radius_(radius), m_MinAngle_(angle0), m_MaxAngle_(angle1) {}
    bool IsClosed() const override { return GetMaxAngle() - GetMinAngle() >= TWOPI; }

    SP_PROPERTY(Real, Radius) = 1.0;
    SP_PROPERTY(Real, MinAngle) = 0.0;
    SP_PROPERTY(Real, MaxAngle) = TWOPI;

    point2d_type xy(Real alpha) const override {
        return point2d_type{m_Radius_ * std::cos(alpha), m_Radius_ * std::sin(alpha)};
    };
};

struct gDisk : public gPlane {
    SP_GEO_ENTITY_HEAD(gPlane, gDisk, Disk);

    explicit gDisk(vector_type const& Nz, vector_type const& Nx, Real radius) : gDisk(Nz, Nx, 0, radius, 0, TWOPI) {}
    explicit gDisk(vector_type const& Nz, vector_type const& Nx, Real min_radius, Real max_radius, Real min_angle,
                   Real max_angle)
        : gPlane(Nz, Nx),
          m_MinRadius_(max_radius),
          m_MaxRadius_(max_radius),
          m_MinAngle_(min_angle),
          m_MaxAngle_(max_angle) {}
    template <typename... Args>
    explicit explicit gDisk(Args&&... args)
        : gDisk(vector_type{0, 0, 1}, vector_type{1, 0, 0}, std::forward<Args>(args)...) {}

    SP_PROPERTY(Real, MinRadius) = 0.0;
    SP_PROPERTY(Real, MaxRadius) = 1.0;
    SP_PROPERTY(Real, MinAngle) = 0.0;
    SP_PROPERTY(Real, MaxAngle) = TWOPI;

    point2d_type xy(Real r, Real alpha) const override {
        return point2d_type{r * std::cos(alpha), r * std::sin(alpha)};
    };
};
}  //    namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_GCIRCLE_H
