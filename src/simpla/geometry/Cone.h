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
namespace simpla {
namespace geometry {

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
