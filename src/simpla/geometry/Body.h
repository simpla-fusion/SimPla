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
};
struct ParametricBody : public Body {
    SP_GEO_ENTITY_ABS_HEAD(Body, ParametricBody)
    virtual point_type xyz(Real u, Real v, Real w) const = 0;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BODY_H
