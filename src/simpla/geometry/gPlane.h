//
// Created by salmon on 17-11-20.
//

#ifndef SIMPLA_GPLANE_H
#define SIMPLA_GPLANE_H

#include "gSurface.h"
namespace simpla {
namespace geometry {
struct gPlane : public gSurface {
    SP_GEO_ENTITY_HEAD(gSurface, gPlane, Plane);
    explicit gPlane(vector_type const& normal, vector_type const& nx)
        : m_Normal_(normal), m_Nx_(nx), m_Ny_(cross(m_Normal_, m_Nx_)){};
    SP_PROPERTY(vector_type, Normal) = {0, 0, 1};
    SP_PROPERTY(vector_type, Nx) = {1, 0, 0};
    SP_PROPERTY(vector_type, Ny) = {0, 1, 0};

    virtual point2d_type xy(Real u, Real v) const { return point2d_type{u, v}; };
    point_type xyz(Real u, Real v) const override {
        auto p = xy(u, v);
        return p[0] * m_Nx_ + p[1] * m_Ny_;
    };
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GPLANE_H
