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
    ~Curve() = default;

    virtual point_type Value(Real u) const = 0;
    virtual bool IsClosed() const { return false; };
    virtual bool IsPeriodic() const { return false; };
    virtual Real Period() const { return INIFITY; };
    virtual Real MinParameter() const { return -INIFITY; }
    virtual Real MaxParameter() const { return INFINITY; }
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CURVE_H
