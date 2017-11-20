//
// Created by salmon on 17-11-7.
//

#ifndef SIMPLA_GWEDGE_H
#define SIMPLA_GWEDGE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>
#include "gBody.h"
namespace simpla {
namespace geometry {

struct gWedge : public gBody {
    SP_GEO_ENTITY_HEAD(gBody, gWedge, Wedge)

    explicit gWedge(vector_type const &extents, Real ltx) : m_Extents_(extents), m_LTX_(ltx) {}

    point_type xyz(Real u, Real v, Real w) const override {
        UNIMPLEMENTED;
        return point_type{0, 0, 0};
    }

    SP_PROPERTY(vector_type, Extents);
    SP_PROPERTY(Real, LTX);
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GWEDGE_H
