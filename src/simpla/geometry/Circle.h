//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_CIRCLE_H
#define SIMPLA_CIRCLE_H

#include <cmath>
#include "Curve.h"
#include "ParametricCurve.h"
namespace simpla {
namespace geometry {

struct Circle : public ParametricCurve {
    SP_GEO_OBJECT_HEAD(Circle, ParametricCurve);

   protected:
    Circle();
    Circle(Circle const &);
    explicit Circle(Axis const &axis, Real radius, Real alpha0 = SP_SNaN, Real alpha1 = SP_SNaN);

   public:
    ~Circle() override;

    /**      ^ axis
     *       |
     *       +----> b
     *       |   /
     *       | /
     *       o
     * @param o
     * @param b
     * @param axis
     * @return
     */
    static std::shared_ptr<Circle> New3(point_type const &o, point_type const &b, vector_type const &axis);

    bool IsClosed() const override { return true; };

    void SetRadius(Real r) { m_radius_ = r; }
    Real GetRadius() const { return m_radius_; }

    point_type xyz(Real alpha) const override {
        return m_axis_.Coordinates(m_radius_ * std::cos(alpha), m_radius_ * std::sin(alpha));
        //        return  m_axis_.o + m_radius_ * std::cos(alpha) * m_axis_.x + m_radius_ * std::sin(alpha) * m_axis_.y;
    };

   protected:
    Real m_radius_ = 1;
};
}  //    namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_CIRCLE_H
