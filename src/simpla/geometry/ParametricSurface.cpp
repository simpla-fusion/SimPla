//
// Created by salmon on 17-10-31.
//

#include "ParametricSurface.h"
#include "GeoAlgorithm.h"
#include "Line.h"
#include "ShapeFunction.h"
namespace simpla {
namespace geometry {
ParametricSurface::ParametricSurface() = default;
ParametricSurface::ParametricSurface(ParametricSurface const &other) = default;
ParametricSurface::ParametricSurface(Axis const &axis) : Surface(axis) {}
ParametricSurface::~ParametricSurface() = default;
bool ParametricSurface::IsClosed() const { return false; };

std::shared_ptr<data::DataNode> ParametricSurface::Serialize() const { return base_type::Serialize(); };
void ParametricSurface::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

box_type ParametricSurface::GetBoundingBox() const { return GetValueRange(); };

point_type ParametricSurface::xyz(point_type const &u) const { return xyz(u[0], u[1]); }
point_type ParametricSurface::uvw(point_type const &x) const { return uvw(x[0], x[1], x[2]); };

}  // namespace geometry
}  // namespace simpla