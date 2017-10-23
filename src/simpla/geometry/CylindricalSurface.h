//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_CYLINDRICALSURFACE_H
#define SIMPLA_CYLINDRICALSURFACE_H

#include <simpla/utilities/Constants.h>
#include <simpla/utilities/macro.h>
#include "Surface.h"
namespace simpla {
namespace geometry {
struct CylindricalSurface : public Surface {
    SP_GEO_OBJECT_HEAD(CylindricalSurface, Surface);

   protected:
    CylindricalSurface() = default;
    CylindricalSurface(CylindricalSurface const &other) = default;  // : Surface(other), m_radius_(other.m_radius_) {}
    CylindricalSurface(std::shared_ptr<Axis> const &axis, Real R, Real phi0 = SP_SNaN, Real phi1 = SP_SNaN,
                       Real z0 = SP_SNaN, Real z1 = SP_SNaN)
        : Surface(axis), m_radius_(R) {
        auto min = GetMinParameter();
        auto max = GetMaxParameter();

        TRY_ASSIGN(min[0], phi0);
        TRY_ASSIGN(max[0], phi1);
        TRY_ASSIGN(min[1], z0);
        TRY_ASSIGN(min[1], z1);

        SetParameterRange(min, max);
    }

   public:
    ~CylindricalSurface() override = default;

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(true, false); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(true, false); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{TWOPI, SP_INFINITY}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{0, -SP_INFINITY}; }
    nTuple<Real, 2> GetMaxParameter() const override { return nTuple<Real, 2>{TWOPI, SP_INFINITY}; }

    void SetRadius(Real r) { m_radius_ = r; }
    Real GetRadius() const { return m_radius_; }

    /**
     *
     * @param u  \phi
     * @param v  z
     * @return
     */
    point_type Value(Real u, Real v) const override {
        return m_axis_->Coordinates(m_radius_ * std::cos(u), m_radius_ * std::sin(u), v);
    };

   private:
    Real m_radius_ = 1.0;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_CYLINDRICALSURFACE_H
