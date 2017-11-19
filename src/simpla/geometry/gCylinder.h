gCylinderSurface//
// Created by salmon on 17-7-22.
//

#ifndef SIMPLA_GCYLINDRICAL_H
#define SIMPLA_GCYLINDRICAL_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>
#include "Body.h"
#include "Surface.h"
namespace simpla {
namespace geometry {
/**
*  R phi Z
*/
struct gCylinder : public ParametricBody {
    SP_GEO_ENTITY_HEAD(ParametricBody, gCylinder, Cylinder)
    gCylinder(Real radius, Real height) : m_Radius_(radius), m_Height_(height) {}
    SP_PROPERTY(Real, Radius) = 1.0;
    SP_PROPERTY(Real, Height) = 1.0;
    point_type xyz(Real u, Real v, Real w) const override { return point_type{u * std::cos(v), u * std::sin(v), w}; };
};
struct gCylindricalSurface : public ParametricSurface {
    SP_GEO_ENTITY_HEAD(ParametricSurface, gCylindricalSurface, gCylindricalSurface)
    gCylindricalSurface(Real radius) : m_Radius_(radius) {}
    SP_PROPERTY(Real, Radius) = 1.0;
    point_type xyz(Real v, Real w) const override {
        return point_type{m_Radius_ * std::cos(v), m_Radius_ * std::sin(v), w};
    };
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_GCYLINDRICAL_H
