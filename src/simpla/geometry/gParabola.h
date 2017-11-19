//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_PARABOLA_H
#define SIMPLA_PARABOLA_H

#include <simpla/data/Serializable.h>
#include "gConic.h"

namespace simpla {
namespace geometry {

struct gParabola : public gConic {
    SP_GEO_ENTITY_HEAD(gConic, gParabola, Parabola);
    explicit gParabola(Real focal) : m_Focal_(focal) {}
    SP_PROPERTY(Real, Focal);
    point2d_type xy(Real u) const override { return point2d_type{u * u / (4. * m_Focal_), u}; };
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_PARABOLA_H
