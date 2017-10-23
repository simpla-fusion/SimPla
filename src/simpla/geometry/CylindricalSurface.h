//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_CYLINDRICALSURFACE_H
#define SIMPLA_CYLINDRICALSURFACE_H

#include <simpla/utilities/Constants.h>
#include "ParametricSurface.h"
namespace simpla {
namespace geometry {
struct CylindricalSurface : public ParametricSurface {
    SP_GEO_OBJECT_HEAD(CylindricalSurface, ParametricSurface);
    CylindricalSurface() = default;
    CylindricalSurface(CylindricalSurface const &) = default;
    ~CylindricalSurface() override = default;

    template <typename... Args>
    explicit CylindricalSurface(Real R, Args &&... args)
        : ParametricSurface(std::forward<Args>(args)...), m_radius_(R) {}

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(true, false); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(true, false); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{TWOPI, SP_INFINITY}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{0, -SP_INFINITY}; }
    nTuple<Real, 2> GetMaxParameter() const override { return nTuple<Real, 2>{TWOPI, SP_INFINITY}; }

    void SetRadius(Real r) { m_radius_ = r; }
    Real GetRadius() const { return m_radius_; }

    point_type Value(Real u, Real v) const override {
        return m_origin_ + m_radius_ * std::cos(u) * m_x_axis_ + m_radius_ * std::sin(u) * m_y_axis_ + v * m_z_axis_;
    };

   private:
    Real m_radius_ = 1.0;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_CYLINDRICALSURFACE_H
