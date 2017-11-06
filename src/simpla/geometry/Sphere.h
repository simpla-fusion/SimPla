//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_SPHERE_H
#define SIMPLA_SPHERE_H

#include <simpla/SIMPLA_config.h>
#include "PrimitiveShape.h"
namespace simpla {
namespace geometry {

struct Sphere : public PrimitiveShape {
    SP_GEO_OBJECT_HEAD(Sphere, PrimitiveShape)

   protected:
    Sphere(Axis const &axis, Real radius, Real phi0, Real phi1, Real theta0, Real theta1);

   public:
    point_type xyz(Real r, Real phi, Real theta) const override {
        Real cos_theta = std::cos(theta);
        return m_axis_.Coordinates(r * cos_theta * std::cos(phi), r * cos_theta * std::sin(phi), r * std::sin(theta));
    };
    point_type uvw(Real x, Real y, Real z) const override { return point_type{x, y, z}; };

    void SetRadius(Real);
    Real GetRadius() const;

    bool CheckIntersection(point_type const &x, Real tolerance) const override;
    bool CheckIntersection(box_type const &, Real tolerance) const override;

   private:
    Real m_radius_ = 1.0;
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_SPHERE_H
