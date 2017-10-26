//
// Created by salmon on 17-10-19.
//

#include "Surface.h"
#include <simpla/SIMPLA_config.h>
#include "Curve.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

std::shared_ptr<data::DataNode> Surface::Serialize() const { return base_type::Serialize(); };
void Surface::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

PointsOnSurface::PointsOnSurface() = default;
PointsOnSurface::~PointsOnSurface() = default;
PointsOnSurface::PointsOnSurface(std::shared_ptr<const Surface> const &surf) : m_surface_(surf) {}
std::shared_ptr<data::DataNode> PointsOnSurface::Serialize() const { return base_type::Serialize(); };
void PointsOnSurface::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<const Surface> PointsOnSurface::GetBasisSurface() const { return m_surface_; }
void PointsOnSurface::SetBasisSurface(std::shared_ptr<const Surface> const &c) { m_surface_ = c; }

void PointsOnSurface::PutUV(nTuple<Real, 2> uv) { m_data_.push_back(std::move(uv)); }
nTuple<Real, 2> PointsOnSurface::GetUV(size_type i) const { return m_data_[i]; }
point_type PointsOnSurface::GetPoint(size_type i) const { return m_surface_->Value(GetUV(i)); }
size_type PointsOnSurface::size() const { return m_data_.size(); }
std::vector<nTuple<Real, 2>> const &PointsOnSurface::data() const { return m_data_; }
std::vector<nTuple<Real, 2>> &PointsOnSurface::data() { return m_data_; }
}  // namespace geometry
}  // namespace simpla