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
namespace simpla {
namespace geometry {
struct PointsOnCurve;
/**
 *  R phi Z
 */
struct Cylinder : public PrimitiveShape {
    SP_GEO_OBJECT_HEAD(Cylinder, PrimitiveShape)
   public:
    point_type xyz(Real u, Real v, Real w) const override;
    point_type uvw(Real x, Real y, Real z) const override;

    Real GetRadius() const { return m_radius_; }
    void SetRadius(Real r) { m_radius_ = r; }
    Real GetHeight() const { return m_height_; }
    void SetHeight(Real h) { m_height_ = h; }

   protected:
    Real m_radius_ = 1.0;
    Real m_height_ = 1.0;
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICAL_H
