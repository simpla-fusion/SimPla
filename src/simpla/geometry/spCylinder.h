//
// Created by salmon on 17-7-22.
//

#ifndef SIMPLA_CYLINDRICAL_H
#define SIMPLA_CYLINDRICAL_H

#include <simpla/utilities/Constants.h>
#include <simpla/utilities/macro.h>
#include "simpla/SIMPLA_config.h"

#include "Body.h"
#include "GeoObject.h"
#include "Shape.h"
#include "Surface.h"

namespace simpla {
namespace geometry {
/**
*  R phi Z
*/
struct spCylinder : public Shape {
    SP_SHAPE_HEAD(Shape, spCylinder, Cylinder)

    spCylinder(Real radius, Real height);

    Real GetRadius() const { return m_radius_; }
    void SetRadius(Real const &a) { m_radius_ = a; }
    Real GetHeight() const { return m_height_; }
    void SetHeight(Real const &a) { m_height_ = a; }
    Real GetAngle() const { return m_angle_; }
    void SetAngle(Real const &a) { m_angle_ = a; }

    point_type xyz(Real u, Real v, Real w) const {
        return point_type{u * m_radius_ * std::cos(v * m_angle_), u * m_radius_ * std::sin(v * m_angle_),
                          w * m_height_};
    };

   private:
    Real m_radius_;
    Real m_height_;
    Real m_angle_ = TWOPI;
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICAL_H
