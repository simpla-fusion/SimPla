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
#include "PrimitiveShape.h"
#include "Surface.h"
#include "ShapeBox.h"

namespace simpla {
namespace geometry {
/**
*  R phi Z
*/
struct Cylinder : public PrimitiveShape {
    SP_GEO_OBJECT_HEAD(Cylinder, PrimitiveShape)
   protected:
    explicit Cylinder(Axis const &, Real radius, Real height, Real angle = TWOPI);
    explicit Cylinder(Real radius, Real height, Real angle = TWOPI);

   public:
    template <typename... Args>
    static std::shared_ptr<GeoObject> MakeBox(box_type const &b, Args &&... args) {
        return ShapeBox<this_type>::New(b, std::forward<Args>(args)...);
    }
    point_type xyz(Real u, Real v, Real w) const override;
    point_type uvw(Real x, Real y, Real z) const override;

    Real GetRadius() const { return m_radius_; }
    void SetRadius(Real const &a) { m_radius_ = a; }

    Real GetHeight() const { return m_height_; }
    void SetHeight(Real const &a) { m_height_ = a; }

    Real GetAngle() const { return m_angle_; }
    void SetAngle(Real const &a) { m_angle_ = a; }

   private:
    Real m_radius_;
    Real m_height_;
    Real m_angle_;
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICAL_H
