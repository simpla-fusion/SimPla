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

   protected:
    SphericalSurface() = default;
    SphericalSurface(SphericalSurface const &other) = default;  //: Surface(other), m_radius_(other.m_radius_) {}
    SphericalSurface(std::shared_ptr<Axis> const &axis, Real radius) : Surface(axis), m_radius_(radius) {}

   public:
    ~SphericalSurface() override = default;

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(true, true); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(true, true); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{TWOPI, TWOPI}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{0, -PI / 2}; }
    nTuple<Real, 2> GetMaxParameter() const override { return nTuple<Real, 2>{TWOPI, PI / 2}; }

    void SetRadius(Real r) { m_radius_ = r; }
    Real GetRadius() const { return m_radius_; }

    point_type Value(Real phi, Real theta) const override {
        Real cos_theta = std::cos(theta);
        return m_axis_->Coordinates(m_radius_ * cos_theta * std::cos(phi), m_radius_ * cos_theta * std::sin(phi),
                                    m_radius_ * std::sin(theta));
        //        return m_axis_.o + m_radius_ * std::cos(theta) * (std::cos(phi) * m_axis_.x + std::sin(phi) *
        //        m_axis_.y) +
        //               m_radius_ * std::sin(theta) * m_axis_.z;
    };

   private:
    Real m_radius_ = 1.0;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_SPHERICALSURFACE_H
