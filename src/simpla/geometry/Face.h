//
// Created by salmon on 17-11-14.
//

#ifndef SIMPLA_FACE_H
#define SIMPLA_FACE_H

#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"
#include "PrimitiveShape.h"
namespace simpla {
namespace geometry {
struct Face : public PrimitiveShape {
    SP_GEO_ABS_OBJECT_HEAD(Face, PrimitiveShape);

   public:
    virtual point2d_type xy(Real u, Real v) const = 0;
    virtual point2d_type uv(Real x, Real y) const = 0;
    point_type xyz(Real u, Real v, Real w) const override {
        auto p = xy(u, v);
        return m_axis_.xyz(p[0], p[1], 0);
    }
    point_type uvw(Real x, Real y, Real z) const override {
        auto p = m_axis_.uvw(x, y, z);
        auto q = uv(p[0], p[1]);
        return point_type{q[0], q[1], 0};
    }
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_FACE_H
