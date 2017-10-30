//
// Created by salmon on 17-10-18.
//
#include "Body.h"
#include "Curve.h"
#include "Surface.h"
namespace simpla {
namespace geometry {
Body::Body() = default;
Body::Body(Body const &other) = default;
Body::Body(Axis const &axis) : GeoObject(axis) {}
Body::~Body() = default;
std::shared_ptr<data::DataNode> Body::Serialize() const { return base_type::Serialize(); };
void Body::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

}  // namespace geometry
}  // namespace simpla