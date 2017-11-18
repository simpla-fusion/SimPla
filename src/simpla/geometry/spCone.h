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
    explicit spCone(Real angle, Real radius);

    point_type xyz(Real l, Real phi, Real theta) const {
        Real r = l * std::sin(theta);
        return m_axis_.xyz(r * std::cos(phi), r * std::sin(phi), l * std::cos(theta));
    }

    void SetAngle(Real theta) { m_angle_ = theta; }
    Real GetAngle() const { return m_angle_; }
    void SetRadius(Real r) { m_radius_ = r; }
    Real GetRadius() const { return m_radius_; }

   protected:
    Real m_angle_ = PI / 4;
    Real m_radius_ = 1.0;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CONE_H
