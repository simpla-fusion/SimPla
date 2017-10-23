//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_CONICALSURFACE_H
#define SIMPLA_CONICALSURFACE_H

#include <simpla/utilities/Constants.h>
#include "Surface.h"
namespace simpla {
namespace geometry {
struct ConicalSurface : public Surface {
    SP_GEO_OBJECT_HEAD(ConicalSurface, Surface);

   protected:
    ConicalSurface() = default;
    ConicalSurface(ConicalSurface const &other) = default;
    //  : Surface(other), m_radius_(other.m_radius_), m_angle_(other.m_angle_) {}
    ConicalSurface(Axis const &axis, Real R, Real Ang) : Surface(axis), m_radius_(R), m_angle_(Ang) {}

   public:
    ~ConicalSurface() override = default;

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(true, false); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(true, false); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{TWOPI, SP_INFINITY}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{0, -SP_INFINITY}; }
    nTuple<Real, 2> GetMaxParameter() const override { return nTuple<Real, 2>{TWOPI, SP_INFINITY}; }

    void SetRadius(Real r) { m_radius_ = r; }
    Real GetRadius() const { return m_radius_; }
    void SetAngle(Real r) { m_angle_ = r; }
    Real GetAngle() const { return m_angle_; }

    point_type Value(Real u, Real v) const override {
        return m_axis_.o + (m_radius_ + v * std::sin(m_angle_)) * (std::cos(u) * m_axis_.x + std::sin(u) * m_axis_.y) +
               v * std::cos(m_angle_) * m_axis_.z;
    };

   private:
    Real m_radius_ = 1.0;
    Real m_angle_ = PI / 4;
};

}  // namespace simpla
}  // namespace geometry

#endif  // SIMPLA_CONICALSURFACE_H
