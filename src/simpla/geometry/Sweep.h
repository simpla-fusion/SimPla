//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_SWEPTBODY_H
#define SIMPLA_SWEPTBODY_H
#include <simpla/algebra/nTuple.ext.h>
#include <simpla/utilities/Constants.h>
#include "Body.h"
#include "Curve.h"
#include "PrimitiveShape.h"
#include "Surface.h"
namespace simpla {
namespace geometry {

struct Sweep : public PrimitiveShape {
    SP_GEO_ABS_OBJECT_HEAD(Sweep, PrimitiveShape);

   protected:
    explicit Sweep(std::shared_ptr<const GeoObject> const &s, std::shared_ptr<const Curve> const &c);

   public:
    std::shared_ptr<const GeoObject> GetBasisObject() const { return m_basis_obj_; }
    std::shared_ptr<const Curve> GetCurve() const { return m_curve_; }

   private:
    std::shared_ptr<const GeoObject> m_basis_obj_;
    std::shared_ptr<const Curve> m_curve_;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SWEPTBODY_H
