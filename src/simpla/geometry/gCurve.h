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
};

struct gCurve2D : public gCurve {
    SP_GEO_ENTITY_ABS_HEAD(gCurve, gCurve2D)
    explicit gCurve2D(vector_type const& normal, vector_type const& nx)
        : m_Normal_(normal), m_Nx_(nx), m_Ny_(cross(m_Normal_, m_Nx_)){};
    SP_PROPERTY(vector_type, Nx) = {1, 0, 0};
    SP_PROPERTY(vector_type, Ny) = {0, 1, 0};
    SP_PROPERTY(vector_type, Normal) = {0, 0, 1};

    void Update() { m_Ny_ = cross(m_Normal_, m_Nx_); }
    virtual point2d_type xy(Real u) const = 0;
    point_type xyz(Real u) const override {
        auto p = xy(u);
        return p[0] * m_Nx_ + p[1] * m_Ny_;
    };
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CURVE_H
