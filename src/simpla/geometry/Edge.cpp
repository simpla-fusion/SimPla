//
// Created by salmon on 17-11-17.
//

#include "Edge.h"
#include "Curve.h"

namespace simpla {
namespace geometry {
Edge::Edge() = default;
Edge::Edge(Edge const &) = default;
Edge::~Edge() = default;

Edge::Edge(Axis const &axis, std::shared_ptr<const Curve> const &curve, Real u_min, Real u_max)
    : Edge(axis, curve, std::tuple<Real, Real>{u_min, u_max}){};
Edge::Edge(Axis const &axis, std::shared_ptr<const Curve> const &curve, std::tuple<Real, Real> const &range)
    : GeoObject(axis), m_curve_(curve), m_range_{range} {};
void Edge::SetCurve(std::shared_ptr<const Curve> const &s) { m_curve_ = s; }
std::shared_ptr<const Curve> Edge::GetCurve() const { return m_curve_; }
void Edge::SetParameterRange(Real umin, Real umax) { m_range_ = std::tie(umin, umax); };
std::tuple<Real, Real> const &Edge::GetParameterRange() const { return m_range_; };
void Edge::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_curve_ = Curve::New(cfg->Get("Curve"));
    std::get<0>(m_range_) = cfg->GetValue("MinU", std::get<0>(m_range_));
    std::get<1>(m_range_) = cfg->GetValue("MaxU", std::get<1>(m_range_));
};
std::shared_ptr<simpla::data::DataEntry> Edge::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Curve", m_curve_->Serialize());
    res->SetValue("MinU", std::get<0>(m_range_));
    res->SetValue("MaxU", std::get<1>(m_range_));
    return res;
};
}  // namespace geometry
}  // namespace simpla