//
// Created by salmon on 17-7-22.
//

#ifndef SIMPLA_CYLINDRICAL_H
#define SIMPLA_CYLINDRICAL_H

#include <simpla/utilities/Constants.h>
#include "simpla/SIMPLA_config.h"

#include "Body.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Cylindrical : public Body {
    SP_GEO_OBJECT_HEAD(Cylindrical, Body)

   protected:
    Cylindrical() = default;
    explicit Cylindrical(std::shared_ptr<Axis> const &axis) : Body(axis) {}

   public:
    ~Cylindrical() override = default;

    bool CheckInside(point_type const &x, Real tolerance) const override { return true; }

    std::tuple<bool, bool, bool> IsClosed() const override { return std::make_tuple(false, true, false); };
    std::tuple<bool, bool, bool> IsPeriodic() const override { return std::make_tuple(false, true, false); };
    nTuple<Real, 3> GetPeriod() const override { return nTuple<Real, 3>{SP_INFINITY, TWOPI, SP_INFINITY}; };
    nTuple<Real, 3> GetMinParameter() const override { return nTuple<Real, 3>{0, 0, -SP_INFINITY}; }
    nTuple<Real, 3> GetMaxParameter() const override { return nTuple<Real, 3>{SP_INFINITY, TWOPI, -SP_INFINITY}; }
    /**
     *
     * @param u R
     * @param v phi
     * @param w Z
     * @return
     */
    point_type Value(Real u, Real v, Real w) const override {
        return m_axis_->Coordinates(u * std::cos(v), u * std::sin(v), w);
    };
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICAL_H
