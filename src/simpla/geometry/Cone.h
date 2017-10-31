//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_CONE_H
#define SIMPLA_CONE_H

#include <simpla/utilities/Constants.h>
#include "simpla/SIMPLA_config.h"

#include "Body.h"
#include "GeoObject.h"
#include "ParametricBody.h"
#include "ParametricSurface.h"

namespace simpla {
namespace geometry {

struct ConicalSurface : public ParametricSurface {
    SP_GEO_OBJECT_HEAD(ConicalSurface, ParametricSurface);

   protected:
    ConicalSurface() = default;
    ConicalSurface(ConicalSurface const &other) = default;
    ConicalSurface(Axis const &axis, Real radius, Real semi_angle, Real phi0 = SP_SNaN, Real phi1 = SP_SNaN,
                   Real z0 = SP_SNaN, Real z1 = SP_SNaN)
        : ParametricSurface(axis), m_radius_(radius), m_semi_angle_(semi_angle) {}

   public:
    ~ConicalSurface() override = default;

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

struct Cone : public ParametricBody {
    SP_GEO_OBJECT_HEAD(Cone, ParametricBody)
    Cone() = default;
    ~Cone() override = default;

   protected:
    explicit Cone(Axis const &axis, Real angle) : ParametricBody(axis), m_semi_angle_(angle) {}

   public:
    void SetSemiAngle(Real a) { m_semi_angle_ = a; }
    Real GetSemiAngle() const { return m_semi_angle_; }

    /**
     *
     * @param theta R
     * @param v phi
     * @param w theta
     * @return
     */
    point_type xyz(Real R, Real theta, Real v) const override {
        Real r = (R + v * std::sin(m_semi_angle_));
        return m_axis_.Coordinates(r * std::cos(theta), r * std::sin(theta), v * std::cos(m_semi_angle_));
        //        return m_axis_.o +
        //               (R + v * std::sin(m_semi_angle_)) * (std::cos(theta) * m_axis_.x + std::sin(theta) * m_axis_.y)
        //               +
        //               v * std::cos(m_semi_angle_) * m_axis_.z;
    };
    //    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   private:
    Real m_semi_angle_ = PI / 4;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CONE_H
