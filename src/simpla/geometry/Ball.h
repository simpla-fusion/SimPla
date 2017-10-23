//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_BALL_H
#define SIMPLA_BALL_H

#include "Body.h"
namespace simpla {
namespace geometry {
class Ball : public Body {
    SP_GEO_OBJECT_HEAD(Cylindrical, Body)
    Cylindrical() = default;
    ~Cylindrical() override = default;

   protected:
    Cylindrical(Axis const &axis) : Body(axis) {}

   public:
    box_type GetBoundingBox() const override {
        box_type b;
        std::get<0>(b) = m_axis_.o - SP_INFINITY;
        std::get<1>(b) = m_axis_.o + SP_INFINITY;
        return std::move(b);
    };

    bool CheckInside(point_type const &x, Real tolerance) const override { return true; }

    /**
     *
     * @param u R
     * @param v phi
     * @param w Z
     * @return
     */
    point_type Value(Real u, Real v, Real w) const override {
        return m_axis_.o + u * std::cos(v) * m_axis_.x + u * std::sin(v) * m_axis_.y + w * m_axis_.z;
    };
};
}  // namespace simpla {
}  // namespace simpla {

#endif  // SIMPLA_BALL_H
