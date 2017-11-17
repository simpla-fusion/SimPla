//
// Created by salmon on 17-11-17.
//

#include "Edge.h"
namespace simpla {
namespace geometry {
Edge::Edge() = default;
Edge::~Edge() = default;
Edge::Edge(Edge const &other) = default;
Edge::Edge(std::shared_ptr<const Curve> const &curve, Real u_min, Real u_max)
    : m_curve_(curve), m_range_{u_min, u_max} {};
void Edge::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) { base_type::Deserialize(cfg); };
std::shared_ptr<simpla::data::DataEntry> Edge::Serialize() const { return base_type::Serialize(); };
}  // namespace geometry
}  // namespace simpla