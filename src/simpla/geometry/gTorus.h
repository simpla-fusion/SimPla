//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_TOROIDAL_H
#define SIMPLA_TOROIDAL_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>
#include <simpla/utilities/SPDefines.h>
#include <simpla/utilities/macro.h>

#include "Body.h"

namespace simpla {
namespace geometry {
struct gTorus : public ParametricBody {
    SP_GEO_ENTITY_HEAD(ParametricBody, gTorus, Torus)

    explicit gTorus(Real major_radius, Real minor_radius, Real min_major_angle = 0, Real max_major_angle = TWOPI,
                    Real min_minor_angle = 0, Real max_minor_angle = TWOPI)
        : m_MinorRadius_(minor_radius),
          m_MajorRadius_(major_radius),
          m_MinMajorAngle_(min_major_angle),
          m_MaxMajorAngle_(max_major_angle),
          m_MinMinorAngle_(min_minor_angle),
          m_MaxMinorAngle_(max_minor_angle) {}

    SP_PROPERTY(Real, MajorRadius);
    SP_PROPERTY(Real, MinorRadius);
    SP_PROPERTY(Real, MaxMajorAngle);
    SP_PROPERTY(Real, MinMajorAngle);
    SP_PROPERTY(Real, MinMinorAngle);
    SP_PROPERTY(Real, MaxMinorAngle);
    point_type xyz(Real phi, Real theta, Real r) const override {
        Real R = (m_MajorRadius_ + r * std::cos(theta));
        return point_type{R * std::cos(phi), R * std::sin(phi), r * std::sin(theta)};
    };
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_TOROIDAL_H
