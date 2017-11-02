//
// Created by salmon on 17-11-1.
//

#include "BoundedCurve.h"

namespace simpla {
namespace geometry {

BoundedCurve::BoundedCurve() = default;
BoundedCurve::BoundedCurve(BoundedCurve const &other) = default;
BoundedCurve::~BoundedCurve() = default;
BoundedCurve::BoundedCurve(Axis const &axis) : Curve(axis){};

void BoundedCurve::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> BoundedCurve::Serialize() const { return base_type::Serialize(); }

}  // namespace geometry{
}  // namespace simpla{