//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_GCONE_H
#define SIMPLA_GCONE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>
#include "gBody.h"
namespace simpla {
namespace geometry {

struct gCone : public gBody {
    SP_GEO_ENTITY_HEAD(gBody, gCone, Cone)
    explicit gCone(Real angle, Real radius) : m_Angle_(angle), m_Radius_(radius) {}
    point_type xyz(Real l, Real phi, Real theta) const override {
        Real r = l * std::sin(theta);
        return point_type{r * std::cos(phi), r * std::sin(phi), l * std::cos(theta)};
    }
    SP_PROPERTY(Real, Angle);
    SP_PROPERTY(Real, Radius) = 1.0;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GCONE_H
