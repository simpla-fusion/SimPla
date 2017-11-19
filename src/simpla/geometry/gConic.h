//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_GCONIC_H
#define SIMPLA_GCONIC_H
#include "Curve.h"
namespace simpla {
namespace geometry {
struct gConic : public ParametricCurve2D {
    SP_SERIALIZABLE_HEAD(ParametricCurve2D, gConic)
    gConic() = default;
    gConic(gConic const &) = default;
    ~gConic() override = default;
};
}  //  namespace geometry{
}  // namespace simpla
#endif  // SIMPLA_GCONIC_H
