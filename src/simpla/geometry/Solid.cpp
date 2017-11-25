//
// Created by salmon on 17-11-18.
//

#include "Solid.h"
#include "gBody.h"
namespace simpla {
namespace geometry {
Solid::Solid(Axis const &axis, std::shared_ptr<const gBody> const &body, box_type const &b)
    : GeoObjectHandle(axis, body, b) {};
void Solid::SetBody(std::shared_ptr<const gBody> const &s) {
    GeoObjectHandle::SetBasisGeometry(std::dynamic_pointer_cast<const GeoEntity>(s));
}
std::shared_ptr<const gBody> Solid::GetBody() const {
    return std::dynamic_pointer_cast<const gBody>(GeoObjectHandle::GetBasisGeometry());
}

}  // namespace geometry
}  // namespace simpla