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

   protected:
    explicit spCircle(Real radius);

   public:
    point_type xyz(Real alpha, Real r, Real z) const {
        return point_type{r * std::cos(alpha), r * std::sin(alpha), z};
    };
    point_type uvw(Real x, Real y, Real z) const { return point_type{std::atan2(y, x), std::hypot(x, y), z}; }

   private:
    Real m_radius_ = 1;
};
}  //    namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_CIRCLE_H
