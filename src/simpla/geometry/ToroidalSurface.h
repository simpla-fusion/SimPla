//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_TOROIDALSURFACE_H
#define SIMPLA_TOROIDALSURFACE_H

#include <simpla/utilities/Constants.h>
#include "Surface.h"

namespace simpla {
namespace geometry {
struct ToroidalSurface : public Surface {
    SP_GEO_OBJECT_HEAD(ToroidalSurface, Surface);

   protected:
    ToroidalSurface() = default;
    ToroidalSurface(ToroidalSurface const &) = default;
    ToroidalSurface(Axis const &axis, Real major_radius, Real minor_radius)
        : Surface(axis), m_major_radius_(major_radius), m_minor_radius_(minor_radius) {}

   public:
    ~ToroidalSurface() override = default;

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(true, true); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(true, true); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{TWOPI, TWOPI}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{0, 0}; }
    nTuple<Real, 2> GetMaxParameter() const override { return nTuple<Real, 2>{TWOPI, TWOPI}; }

    void GetMajorRadius(Real r) { m_major_radius_ = r; }
    void GetMinorRadius(Real r) { m_minor_radius_ = r; }
    Real GetMajorRadius() const { return m_major_radius_; }
    Real GetMinorRadius() const { return m_minor_radius_; }

    point_type Value(Real u, Real v) const override {
        return m_axis_.o +
               (m_major_radius_ + m_minor_radius_ * std::cos(v)) * (std::cos(u) * m_axis_.x + std::sin(u) * m_axis_.y) +
               m_minor_radius_ * std::sin(v) * m_axis_.z;
    };

   protected:
    Real m_major_radius_ = 1;
    Real m_minor_radius_ = 1;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_TOROIDALSURFACE_H
