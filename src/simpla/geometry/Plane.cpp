//
// Created by salmon on 17-10-20.
//

#include "Plane.h"
#include "GeoAlgorithm.h"
#include "Line.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Plane)
SP_DEF_SHAPE_FUNCTION_PARA_VALUE_RANGE(Plane)
SP_DEF_PARA_VALUE_RANGE(Plane)

Plane::Plane() = default;
Plane::Plane(Plane const &) = default;
Plane::Plane(Axis const &axis) : ParametricSurface(axis) {}
Plane::Plane(point_type const &o, vector_type const &x, vector_type const &y) : Plane(Axis(o, x, y)) {}
Plane::~Plane() = default;

void Plane::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Plane::Serialize() const { return base_type::Serialize(); }
std::shared_ptr<PolyPoints> Plane::Intersection(std::shared_ptr<const Curve> const &g, Real tolerance) const {
    std::shared_ptr<PolyPoints> res = nullptr;
    if (auto line = std::dynamic_pointer_cast<const Line>(g)) {
    } else {
        res = g->Intersection(std::dynamic_pointer_cast<const Plane>(shared_from_this()), tolerance);
    }
}

std::shared_ptr<Curve> Plane::Intersection(std::shared_ptr<const Surface> const &g, Real tolerance) const {}
bool Plane::TestIntersection(point_type const &x, Real tolerance) const override {
    return std::abs(m_axis_.uvw(x)[2]) < tolerance;
}
bool Plane::TestIntersection(box_type const &, Real tolerance) const override {}
}  // namespace geometry{
}  // namespace simpla{impla