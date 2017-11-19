//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_GBOX_H
#define SIMPLA_GBOX_H

#include <simpla/SIMPLA_config.h>
#include "Body.h"
namespace simpla {
namespace geometry {

struct gBox : public ParametricBody {
    SP_GEO_ENTITY_HEAD(GeoEntity, gBox, Box)
    explicit gBox(vector_type const &extents) : m_Extents_(extents) {}
    point_type xyz(Real u, Real v, Real w) const override { return point_type{u, v, w}; }
    SP_PROPERTY(vector_type, Extents);
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GBOX_H
