//
// Created by salmon on 17-10-18.
//

#ifndef SIMPLA_BODY_H
#define SIMPLA_BODY_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTuple.h>
#include <memory>
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Body : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Body, GeoObject);
    Body() = default;
    Body(Body const &) = default;
    ~Body() override = default;

    virtual std::tuple<bool, bool, bool> IsClosed() const { return std::make_tuple(false, false, false); };
    virtual std::tuple<bool, bool, bool> IsPeriodic() const { return std::make_tuple(false, false, false); };
    virtual nTuple<Real, 3> GetPeriod() const { return nTuple<Real, 3>{SP_INFINITY, SP_INFINITY, SP_INFINITY}; };
    virtual nTuple<Real, 3> GetMinParameter() const {
        return nTuple<Real, 3>{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY};
    }
    virtual nTuple<Real, 3> GetMaxParameter() const { return nTuple<Real, 3>{SP_INFINITY, SP_INFINITY, -SP_INFINITY}; }

    virtual point_type Value(Real u, Real v, Real w) const = 0;
    point_type Value(nTuple<Real, 3> const &u) const { return Value(u[0], u[1], v[2]); };
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BODY_H
