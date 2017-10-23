//
// Created by salmon on 17-10-23.
//

#include "Sphere.h"
namespace simpla {
namespace geometry {

std::shared_ptr<simpla::data::DataNode> Sphere::Serialize() const { return base_type::Serialize(); };
void Sphere::Deserialize(std::shared_ptr<data::DataNode> const& cfg) { base_type::Deserialize(cfg); }
}  // namespace geometry {
}  // namespace simpla {
