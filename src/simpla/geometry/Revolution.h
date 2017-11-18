//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_REVOLUTIONBODY_H
#define SIMPLA_REVOLUTIONBODY_H
#include <simpla/utilities/Constants.h>
#include "Revolution.h"
#include "Sweep.h"
namespace simpla {
namespace geometry {
struct Curve;
struct Surface;
struct Revolution : public GeoObject {
    SP_GEO_OBJECT_HEAD(GeoObject, Revolution);

   protected:
    explicit Revolution(Axis const &axis, std::shared_ptr<const GeoObject> const &s, Real angele = TWOPI);
    explicit Revolution(std::shared_ptr<const GeoObject> const &s, Real angele = TWOPI);

   public:
    std::shared_ptr<const GeoObject> GetBasisObject() const { return m_basis_obj_; }
    Real GetAngle() const { return m_angle_; }

   private:
    std::shared_ptr<const GeoObject> m_basis_obj_;
    Real m_angle_ = TWOPI;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_REVOLUTIONBODY_H
