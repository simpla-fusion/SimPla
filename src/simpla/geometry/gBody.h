//
// Created by salmon on 17-10-18.
//

#ifndef SIMPLA_BODY_H
#define SIMPLA_BODY_H

#include <simpla/SIMPLA_config.h>
#include "GeoEntity.h"

namespace simpla {
namespace geometry {

struct gBody : public GeoEntity {
    SP_GEO_ENTITY_ABS_HEAD(GeoEntity, gBody)
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BODY_H
