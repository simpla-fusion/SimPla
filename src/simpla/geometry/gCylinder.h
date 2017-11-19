//
// Created by salmon on 17-7-22.
//

#ifndef SIMPLA_CYLINDRICAL_H
#define SIMPLA_CYLINDRICAL_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>
#include "Body.h"

namespace simpla {
namespace geometry {
/**
*  R phi Z
*/
struct gCylinder : public ParametricBody {
    SP_GEO_ENTITY_HEAD(ParametricBody, gCylinder, Cylinder)

    gCylinder(Real radius, Real height, Real angle = TWOPI) : m_Radius_(radius), m_Angle_(angle), m_Height_(height) {}

    SP_PROPERTY(Real, Radius);
    SP_PROPERTY(Real, Height);
    SP_PROPERTY(Real, Angle);

    point_type xyz(Real u, Real v, Real w) const override { return point_type{u * std::cos(v), u * std::sin(v), w}; };
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICAL_H
