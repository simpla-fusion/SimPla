//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_SPHERICALSURFACE_H
#define SIMPLA_SPHERICALSURFACE_H

#include <simpla/utilities/Constants.h>
#include "Surface.h"
namespace simpla {
namespace geometry {
struct SphericalSurface : public Surface {
    SP_GEO_OBJECT_HEAD(SphericalSurface, Surface);
    SphericalSurface() = default;
    SphericalSurface(SphericalSurface const &) = default;
    ~SphericalSurface() override = default;

    template <typename... Args>
    explicit SphericalSurface(Real R, Args &&... args) : Surface(std::forward<Args>(args)...), m_radius_(R) {}

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(true, true); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(true, true); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{TWOPI, TWOPI}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{0, -PI / 2}; }
    nTuple<Real, 2> GetMaxParameter() const override { return nTuple<Real, 2>{TWOPI, PI / 2}; }

    void SetRadius(Real r) { m_radius_ = r; }
    Real GetRadius() const { return m_radius_; }

    point_type Value(Real u, Real v) const override {
        return m_axis_.o + m_radius_ * std::cos(v) * (std::cos(u) * m_axis_.x + std::sin(u) * m_axis_.y) +
               m_radius_ * std::sin(v) * m_axis_.z;
    };

   private:
    Real m_radius_ = 1.0;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_SPHERICALSURFACE_H
