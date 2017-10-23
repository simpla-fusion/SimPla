//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_CONE_H
#define SIMPLA_CONE_H

#include <simpla/utilities/Constants.h>
#include "simpla/SIMPLA_config.h"

#include "Body.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Cone : public Body {
    SP_GEO_OBJECT_HEAD(Cone, Body)
    Cone() = default;
    ~Cone() override = default;

   protected:
    explicit Cone(std::shared_ptr<Axis> const &axis, Real angle) : Body(axis), m_semi_angle_(angle) {}

   public:
    bool CheckInside(point_type const &x, Real tolerance) const override { return true; }

    std::tuple<bool, bool, bool> IsClosed() const override { return std::make_tuple(false, true, false); };
    std::tuple<bool, bool, bool> IsPeriodic() const override { return std::make_tuple(false, true, false); };
    nTuple<Real, 3> GetPeriod() const override { return nTuple<Real, 3>{SP_INFINITY, TWOPI, SP_INFINITY}; };
    nTuple<Real, 3> GetMinParameter() const override { return nTuple<Real, 3>{0, 0, -SP_INFINITY}; }
    nTuple<Real, 3> GetMaxParameter() const override { return nTuple<Real, 3>{SP_INFINITY, TWOPI, SP_INFINITY}; }

    void SetSemiAngle(Real a) { m_semi_angle_ = a; }
    Real GetSemiAngle() const { return m_semi_angle_; }

    /**
     *
     * @param theta R
     * @param v phi
     * @param w theta
     * @return
     */
    point_type Value(Real R, Real theta, Real v) const override {
        Real r = (R + v * std::sin(m_semi_angle_));
        return m_axis_->Coordinates(r * std::cos(theta), r * std::sin(theta), v * std::cos(m_semi_angle_));
        //        return m_axis_.o +
        //               (R + v * std::sin(m_semi_angle_)) * (std::cos(theta) * m_axis_.x + std::sin(theta) * m_axis_.y)
        //               +
        //               v * std::cos(m_semi_angle_) * m_axis_.z;
    };

   private:
    Real m_semi_angle_ = PI / 4;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CONE_H
