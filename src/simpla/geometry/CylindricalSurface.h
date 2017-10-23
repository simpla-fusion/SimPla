//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_CYLINDRICALSURFACE_H
#define SIMPLA_CYLINDRICALSURFACE_H

#include <simpla/utilities/Constants.h>
#include "Surface.h"
namespace simpla {
namespace geometry {
struct CylindricalSurface : public Surface {
    SP_GEO_OBJECT_HEAD(CylindricalSurface, Surface);

   protected:
    CylindricalSurface() = default;
    CylindricalSurface(CylindricalSurface const &other) = default;  // : Surface(other), m_radius_(other.m_radius_) {}
    CylindricalSurface(Axis const &axis, Real R) : Surface(axis), m_radius_(R) {}

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
        return m_axis_.o + m_radius_ * std::cos(u) * m_axis_.x + m_radius_ * std::sin(u) * m_axis_.y + v * m_axis_.z;
    };

   private:
    Real m_radius_ = 1.0;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_CYLINDRICALSURFACE_H
