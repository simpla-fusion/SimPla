//
// Created by salmon on 17-10-31.
//

#include "ParametricCurve.h"
namespace simpla {
namespace geometry {
ParametricCurve::ParametricCurve() = default;
ParametricCurve::ParametricCurve(ParametricCurve const &other) = default;
ParametricCurve::~ParametricCurve() = default;
ParametricCurve::ParametricCurve(Axis const &axis) : Curve(axis){};

box_type ParametricCurve::GetBoundingBox() const { return box_type{{0, 0, 0}, {1, 1, 1}}; };
std::shared_ptr<PolyPoints> ParametricCurve::GetBoundaryPoints() const { return nullptr; };
point_type ParametricCurve::xyz(Real u) const { return point_type{SP_SNaN, SP_SNaN, SP_SNaN}; }
point_type ParametricCurve::xyz(point_type const &u) const { return xyz(u[0]); }
point_type ParametricCurve::uvw(point_type const &x) const { return point_type{SP_SNaN, SP_SNaN, SP_SNaN}; }
void ParametricCurve::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> ParametricCurve::Serialize() const { return base_type::Serialize(); }
}
}