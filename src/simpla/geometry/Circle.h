//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_CIRCLE_H
#define SIMPLA_CIRCLE_H

#include "ParametricCurve.h"
namespace simpla {
namespace geometry {

struct Circle : public ParametricCurve {
    SP_GEO_OBJECT_HEAD(Circle, ParametricCurve);

   public:
    Circle() = default;
    Circle(Circle const &other) = default;
    ~Circle() override = default;

    template <typename... Args>
    explicit Circle(Real radius, Args &&... args) : ParametricCurve(std::forward<Args>(args)...), m_radius_(radius) {}

    point_type Value(Real alpha) const override {
        return m_origin_ + m_radius_ * std::cos(alpha) * m_x_axis_ + m_radius_ * std::sin(alpha) * m_y_axis_;
    };
    bool IsClosed() const override { return true; };
    bool IsPeriodic() const override { return true; };
    Real GetPeriod() const override { return TWOPI; };
    Real GetMinParameter() const override { return 0.0; }
    Real GetMaxParameter() const override { return TWOPI; };

    void SetRadius(Real r) { m_radius_ = r; }
    Real GetRadius() const { return m_radius_; }

   protected:
    Real m_radius_ = 1;
};
}  //    namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_CIRCLE_H
