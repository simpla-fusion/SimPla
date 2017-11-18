//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_CONE_H
#define SIMPLA_CONE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>

#include "Shape.h"
namespace simpla {
namespace geometry {

struct spCone : public Shape {
    SP_SHAPE_HEAD(Shape, spCone, Cone)
   protected:
    explicit spCone(Real semi_angle);

   public:
    //    point_type xyz(Real l, Real phi, Real theta) const override {
    //        Real r = l * std::sin(theta);
    //        return m_axis_.xyz(r * std::cos(phi), r * std::sin(phi), l * std::cos(theta));
    //    }
    //    point_type uvw(Real x, Real y, Real z) const override {
    //        UNIMPLEMENTED;
    //        return point_type{x, y, z};
    //    }

    void SetSemiAngle(Real theta) { m_semi_angle_ = theta; }
    Real GetSemiAngle() const { return m_semi_angle_; }

   protected:
    Real m_semi_angle_ = PI / 4;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CONE_H
