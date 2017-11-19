//
// Created by salmon on 17-11-2.
//

#include "PointsOnCurve.h"
#include "Curve.h"
namespace simpla {
namespace geometry {

PointsOnCurve::PointsOnCurve() = default;
PointsOnCurve::PointsOnCurve(PointsOnCurve const &) = default;
PointsOnCurve::PointsOnCurve(Axis const &axis) : base_type(axis){};
PointsOnCurve::~PointsOnCurve() = default;
void PointsOnCurve::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) { base_type::Deserialize(cfg); };
std::shared_ptr<simpla::data::DataEntry> PointsOnCurve::Serialize() const { return base_type::Serialize(); };
void PointsOnCurve::SetCurve(std::shared_ptr<const Curve> const &g) { m_curve_ = g; }
std::shared_ptr<const Curve> PointsOnCurve::GetCurve() const { return m_curve_; }
size_type PointsOnCurve::size() const { return m_data_.size(); };
point_type PointsOnCurve::GetPoint(size_type idx) const { return m_curve_->xyz(m_data_[idx]); };
void PointsOnCurve::push_back(Real u) { m_data_.push_back(u); }

}  // namespace geometry
}  // namespace simpla