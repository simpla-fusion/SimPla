//
// Created by salmon on 17-10-23.
//

#include "Sphere.h"
#include <simpla/utilities/SPDefines.h>
#include "Circle.h"
#include "Curve.h"
#include "Line.h"
namespace simpla {
namespace geometry {

std::shared_ptr<simpla::data::DataNode> Sphere::Serialize() const { return base_type::Serialize(); };
void Sphere::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

int Sphere::CheckOverlap(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> Sphere::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}
}  // namespace geometry {
}  // namespace simpla {
