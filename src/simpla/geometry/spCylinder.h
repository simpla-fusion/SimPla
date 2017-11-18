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
   protected:
    spCylinder(Real radius, Real height);

   public:
    //    point_type xyz(Real u, Real v, Real w) const override;
    //    point_type uvw(Real x, Real y, Real z) const override;

    Real GetRadius() const { return m_radius_; }
    void SetRadius(Real const &a) { m_radius_ = a; }

    Real GetHeight() const { return m_height_; }
    void SetHeight(Real const &a) { m_height_ = a; }

   private:
    Real m_radius_;
    Real m_height_;
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICAL_H
