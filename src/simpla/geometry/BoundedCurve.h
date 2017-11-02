//
// Created by salmon on 17-11-1.
//

#ifndef SIMPLA_BOUNDEDCURVE_H
#define SIMPLA_BOUNDEDCURVE_H

#include "Curve.h"
namespace simpla {
namespace geometry {
class BoundedCurve : public Curve {
    SP_GEO_ABS_OBJECT_HEAD(BoundedCurve, Curve);

   protected:
    BoundedCurve();
    BoundedCurve(BoundedCurve const &);
    explicit BoundedCurve(Axis const &axis);

   public:
    ~BoundedCurve() override;
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_BOUNDEDCURVE_H
