//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_CONICALSURFACE_H
#define SIMPLA_CONICALSURFACE_H

#include <simpla/utilities/Constants.h>
#include <simpla/utilities/macro.h>
#include "Surface.h"
namespace simpla {
namespace geometry {
struct ConicalSurface : public Surface {
    SP_GEO_OBJECT_HEAD(ConicalSurface, Surface);

   protected:
    ConicalSurface() = default;
    ConicalSurface(ConicalSurface const &other) = default;
    ConicalSurface(Axis const &axis, Real radius, Real semi_angle, Real phi0 = SP_SNaN, Real phi1 = SP_SNaN,
                   Real z0 = SP_SNaN, Real z1 = SP_SNaN)
        : Surface(axis), m_radius_(radius), m_semi_angle_(semi_angle) {
        auto min = GetMinParameter();
        auto max = GetMaxParameter();

        TRY_ASSIGN(min[0], phi0);
        TRY_ASSIGN(max[0], phi1);
        TRY_ASSIGN(min[1], z0);
        TRY_ASSIGN(min[1], z1);

        SetParameterRange(min, max);
    }

   public:
    ~ConicalSurface() override = default;

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(true, false); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(true, false); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{TWOPI, SP_INFINITY}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{0, -SP_INFINITY}; }
    nTuple<Real, 2> GetMaxParameter() const override { return nTuple<Real, 2>{TWOPI, SP_INFINITY}; }

    void SetRadius(Real r) { m_radius_ = r; }
    Real GetRadius() const { return m_radius_; }
    void SetSemiAngle(Real r) { m_semi_angle_ = r; }
    Real GetSemiAngle() const { return m_semi_angle_; }

    point_type Value(Real u, Real v) const override {
        Real r = (m_radius_ + v * std::sin(m_semi_angle_));
        return m_axis_.Coordinates(r * std::cos(u), r * std::sin(u), v * std::cos(m_semi_angle_));
    };

   private:
    Real m_radius_ = 1.0;
    Real m_semi_angle_ = PI / 4;
};

}  // namespace simpla
}  // namespace geometry

#endif  // SIMPLA_CONICALSURFACE_H
