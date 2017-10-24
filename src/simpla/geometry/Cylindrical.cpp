//
// Created by salmon on 17-7-22.
//

#include "Cylindrical.h"
namespace simpla {
namespace geometry {

std::shared_ptr<simpla::data::DataNode> Cylindrical::Serialize() const { return base_type::Serialize(); };
void Cylindrical::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

int Cylindrical::CheckOverlap(box_type const &, Real tolerance) const {}
int Cylindrical::FindIntersection(std::shared_ptr<const Curve> const &, std::vector<Real> &, Real tolerance) const {}
}  // namespace geometry {
}  // namespace simpla {
