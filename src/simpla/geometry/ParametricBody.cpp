//
// Created by salmon on 17-10-31.
//

#include "ParametricBody.h"
#include "GeoAlgorithm.h"

namespace simpla {
namespace geometry {
ParametricBody::ParametricBody() = default;
ParametricBody::ParametricBody(ParametricBody const &other) = default;
ParametricBody::ParametricBody(Axis const &axis) : Body(axis) {}
ParametricBody::~ParametricBody() = default;

std::shared_ptr<data::DataNode> ParametricBody::Serialize() const { return base_type::Serialize(); };
void ParametricBody::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
box_type ParametricBody::GetBoundingBox() const { return GetValueRange(); };

point_type ParametricBody::xyz(point_type const &u) const { return xyz(u[0], u[1], u[2]); }
point_type ParametricBody::uvw(point_type const &x) const { return uvw(x[0], x[1], x[2]); };

}  // namespace geometry
}  // namespace simpla