//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_GCIRCLE_H
#define SIMPLA_GCIRCLE_H

#include <simpla/data/Configurable.h>
#include <simpla/utilities/Constants.h>
#include <cmath>
#include "GeoEntity.h"
#include "gCone.h"
#include "gPlane.h"
#include "gSurface.h"
namespace simpla {
namespace geometry {
struct gCircle : public gConic {
    SP_GEO_ENTITY_HEAD(gConic, gCircle, Circle);
    explicit gCircle(Real radius) : m_Radius_(radius) {}
    bool IsClosed() const override { return true; }
    SP_PROPERTY(Real, Radius) = 1.0;

    point2d_type xy(Real alpha) const override {
        return point2d_type{m_Radius_ * std::cos(alpha), m_Radius_ * std::sin(alpha)};
    };
};

struct gDisk : public gPlane {
    SP_GEO_ENTITY_HEAD(gPlane, gDisk, Disk);
    point2d_type xy(Real r, Real alpha) const override {
        return point2d_type{r * std::cos(alpha), r * std::sin(alpha)};
    };
};
}  //    namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_GCIRCLE_H
