//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_CIRCLE_H
#define SIMPLA_CIRCLE_H

#include <cmath>
#include "Shape.h"
namespace simpla {
namespace geometry {

struct spCircle : public Shape {
    SP_SHAPE_HEAD(Shape, spCircle, Circle);

    explicit spCircle(Real radius);

    point2d_type xy(Real alpha) const {
        return point2d_type{m_radius_ * std::cos(alpha), m_radius_ * std::sin(alpha)};
    };
    Real GetRadius() const { return m_radius_; }
    void SetRadius(Real const &a) { m_radius_ = a; }

   private:
    Real m_radius_ = 1;
};
}  //    namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_CIRCLE_H
