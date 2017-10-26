//
// Created by salmon on 17-7-22.
//

#ifndef SIMPLA_CYLINDRICAL_H
#define SIMPLA_CYLINDRICAL_H

#include <simpla/utilities/Constants.h>
#include <simpla/utilities/macro.h>
#include "simpla/SIMPLA_config.h"

#include "Body.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Cylindrical : public Body {
    SP_GEO_OBJECT_HEAD(Cylindrical, Body)

   protected:
    Cylindrical() = default;
    explicit Cylindrical(Axis const &axis, Real r0 = SP_SNaN, Real r1 = SP_SNaN, Real phi0 = SP_SNaN,
                         Real phi1 = SP_SNaN, Real z0 = SP_SNaN, Real z1 = SP_SNaN)
        : Body(axis) {
        auto min = GetMinParameter();
        auto max = GetMaxParameter();
        TRY_ASSIGN(min[0], r0);
        TRY_ASSIGN(max[0], r1);
        TRY_ASSIGN(min[1], phi0);
        TRY_ASSIGN(max[1], phi1);
        TRY_ASSIGN(min[2], z0);
        TRY_ASSIGN(min[2], z1);

        SetParameterRange(min, max);
    }

   public:
    ~Cylindrical() override = default;

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
        return m_axis_.Coordinates(u * std::cos(v), u * std::sin(v), w);
    };

    int CheckOverlap(box_type const &, Real tolerance) const override;
    int FindIntersection(std::shared_ptr<const Curve> const &, std::vector<Real> &, Real tolerance) const override;
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICAL_H
