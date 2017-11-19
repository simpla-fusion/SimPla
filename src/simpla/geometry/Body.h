//
// Created by salmon on 17-10-18.
//

#ifndef SIMPLA_BODY_H
#define SIMPLA_BODY_H

#include <simpla/SIMPLA_config.h>
#include "GeoEntity.h"

namespace simpla {
namespace geometry {

struct Body : public GeoEntity {
    SP_GEO_ENTITY_ABS_HEAD(GeoEntity, Body)
    Body() = default;
};
struct ParametricBody : public Body {
    SP_GEO_ENTITY_ABS_HEAD(Body, ParametricBody)
    ParametricBody()
        : m_MinU_(-SP_INFINITY),
          m_MaxU_(SP_INFINITY),
          m_MinV_(-SP_INFINITY),
          m_MaxV_(SP_INFINITY),
          m_MinW_(-SP_INFINITY),
          m_MaxW_(SP_INFINITY) {}

    virtual point_type xyz(Real u, Real v, Real w) const = 0;

    SP_PROPERTY(Real, MinU);
    SP_PROPERTY(Real, MaxU);
    SP_PROPERTY(Real, MinV);
    SP_PROPERTY(Real, MaxV);
    SP_PROPERTY(Real, MinW);
    SP_PROPERTY(Real, MaxW);
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BODY_H
