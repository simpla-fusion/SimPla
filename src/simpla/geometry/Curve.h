//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CURVE_H
#define SIMPLA_CURVE_H

#include <simpla/utilities/Constants.h>
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Curve : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Curve, GeoObject);
    Curve() = default;
    Curve(Curve const &) = default;
    ~Curve() override = default;

    virtual bool IsClosed() const { return false; };
    virtual bool IsPeriodic() const { return false; };
    virtual Real GetPeriod() const { return SP_INFINITY; };
    virtual Real GetMinParameter() const { return -SP_INFINITY; }
    virtual Real GetMaxParameter() const { return SP_INFINITY; }

    virtual point_type Value(Real u) const = 0;


};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CURVE_H
