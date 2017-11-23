//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CURVE_H
#define SIMPLA_CURVE_H

#include "GeoEntity.h"
namespace simpla {
namespace geometry {
struct gCurve : public GeoEntity {
    SP_GEO_ENTITY_ABS_HEAD(GeoEntity, gCurve)
    virtual bool IsClosed() const { return false; }
    virtual point_type xyz(Real u) const = 0;
    point_type xyz(Real u, Real v, Real w) const override { return xyz(u); };
};

struct gCurve2D : public gCurve {
    SP_GEO_ENTITY_ABS_HEAD(gCurve, gCurve2D)

    virtual point2d_type xy(Real u) const = 0;
    point_type xyz(Real u) const override {
        point_type p{0, 0, 0};
        p = xy(u);
        return std::move(p);
    };
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CURVE_H
