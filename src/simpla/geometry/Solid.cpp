//
// Created by salmon on 17-11-18.
//

#include "Solid.h"
#include "gBody.h"
namespace simpla {
namespace geometry {
Solid::Solid(std::shared_ptr<const GeoEntity> const &body)
    : Solid(std::dynamic_pointer_cast<const gBody>(body), Axis{}, box_type{{0, 0, 0}, {1, 1, 1}}) {}
Solid::Solid(std::shared_ptr<const gBody> const &body, Axis const &axis, Real u_min, Real u_max, Real v_min, Real v_max,
             Real w_min, Real w_max)
    : Solid(body, axis, box_type{{u_min, v_min, w_min}, {u_max, v_max, w_max}}) {}
Solid::Solid(std::shared_ptr<const gBody> const &body, Axis const &axis, point_type const &u_min,
             point_type const &u_max)
    : Solid(body, axis, box_type{u_min, u_max}) {}
Solid::Solid(std::shared_ptr<const gBody> const &body, Axis const &axis, box_type const &b)
    : GeoObjectHandle(body, axis){};
void Solid::SetBody(std::shared_ptr<const gBody> const &s) {
    GeoObjectHandle::SetBasisGeometry(std::dynamic_pointer_cast<const GeoEntity>(s));
}
std::shared_ptr<const gBody> Solid::GetBody() const {
    return std::dynamic_pointer_cast<const gBody>(GeoObjectHandle::GetBasisGeometry());
}

}  // namespace geometry
}  // namespace simpla