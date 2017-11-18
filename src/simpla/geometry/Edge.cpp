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
    : GeoObject(axis), m_curve_(curve), m_range_{u_min, u_max} {};
Edge::Edge(std::shared_ptr<const Curve> const &curve, std::tuple<Real, Real> const &range)
    : GeoObject(axis), m_curve_(curve), m_range_{range} {};

void Edge::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_range_ = cfg->GetValue("ParameterRange", m_range_);
};
std::shared_ptr<simpla::data::DataEntry> Edge::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("ParameterRange", m_range_);
    return res;
};
}  // namespace geometry
}  // namespace simpla