//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_SWEPTBODY_H
#define SIMPLA_SWEPTBODY_H
#include <simpla/algebra/nTuple.ext.h>
#include <simpla/utilities/Constants.h>
#include "Body.h"
#include "Curve.h"
#include "GeoObject.h"
#include "Surface.h"
namespace simpla {
namespace geometry {
struct Face;
struct Edge;
struct Sweep {
   protected:
    explicit Sweep(std::shared_ptr<const GeoObject> const &s, std::shared_ptr<const Curve> const &c);

   public:
    std::shared_ptr<const GeoObject> GetBasisObject() const { return m_basis_obj_; }
    std::shared_ptr<const Curve> GetCurve() const { return m_curve_; }

   private:
    std::shared_ptr<const GeoObject> m_basis_obj_;
    std::shared_ptr<const Curve> m_curve_;
};
std::shared_ptr<GeoObject> make_Sweep(std::shared_ptr<const Edge> const &e0, std::shared_ptr<const Edge> const &e1);
std::shared_ptr<GeoObject> make_Sweep(std::shared_ptr<const Face> const &f0, std::shared_ptr<const Edge> const &e1);
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SWEPTBODY_H
