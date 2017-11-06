//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_CONE_H
#define SIMPLA_CONE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>

#include "PrimitiveShape.h"
namespace simpla {
namespace geometry {

struct Cone : public PrimitiveShape {
    SP_GEO_OBJECT_HEAD(Cone, PrimitiveShape)
   protected:
    explicit Cone(Real semi_angle);
    explicit Cone(Axis const& axis, Real semi_angle);

   public:
    point_type xyz(Real l, Real phi, Real theta) const override {
        Real r = l * std::sin(theta);
        return m_axis_.xyz(r * std::cos(phi), r * std::sin(phi), l * std::cos(theta));
    }
    point_type uvw(Real x, Real y, Real z) const override {
        UNIMPLEMENTED;
        return point_type{x, y, z};
    }

    void SetSemiAngle(Real theta) { m_semi_angle_ = theta; }
    Real GetSemiAngle() const { return m_semi_angle_; }

   protected:
    Real m_semi_angle_ = PI / 4;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CONE_H
